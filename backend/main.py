import os
import re
import json
import pickle
import uuid
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# =========================
# Config (env-tunable)
# =========================
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3:8b")

TOP_K = int(os.getenv("TOP_K", "4"))                      # retrieval depth
CONTEXT_CHAR_LIMIT = int(os.getenv("CONTEXT_CHAR_LIMIT", "2500"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "150"))  # default answer length
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploaded_docs"))
STORE_DIR  = os.getenv("STORE_DIR",  str(BASE_DIR / "store"))
INDEX_PATH = str(Path(STORE_DIR) / "index.faiss")
# --- NEW: dual-index paths for section-aware retrieval ---
SECT_INDEX_PATH = str(Path(STORE_DIR) / "sections.index.faiss")
SECT_META_PATH  = str(Path(STORE_DIR) / "sections.meta.pkl")
CHUNK_INDEX_PATH = str(Path(STORE_DIR) / "chunks.index.faiss")
CHUNK_META_PATH  = str(Path(STORE_DIR) / "chunks.meta.pkl")

# Section-aware retrieval knobs
SECTION_OVERFETCH = int(os.getenv("SECTION_OVERFETCH", "12"))
SECTIONS_FINAL = int(os.getenv("SECTIONS_FINAL", "6"))        # keep this many sections after diversity
CHUNKS_PER_SECTION = int(os.getenv("CHUNKS_PER_SECTION", "2")) # zoom-in per section
GLOBAL_TOP_K = int(os.getenv("GLOBAL_TOP_K", "6"))             # final chunk count (can be >= TOP_K)

META_PATH  = str(Path(STORE_DIR) / "meta.pkl")
DOCS_PATH  = str(Path(STORE_DIR) / "docs.json")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DIVERSITY_SIM_THRESHOLD = float(os.getenv("DIVERSITY_SIM_THRESHOLD", "0.95"))
OVERFETCH_MULT = int(os.getenv("OVERFETCH_MULT", "5"))

# Heuristic triggers for overview questions
OVERVIEW_TRIGGERS = tuple(
    s.strip() for s in os.getenv(
        "OVERVIEW_TRIGGERS",
        "what is;what’s;whats;overview;summary;high level;broadly;explain the paper;describe the paper"
    ).split(";")
    if s.strip()
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STORE_DIR, exist_ok=True)

# =========================
# FastAPI app
# =========================
app = FastAPI(docs_url="/api", redoc_url=None, openapi_url="/openapi.json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Globals (in-memory)
# =========================
emb_model: SentenceTransformer | None = None
faiss_index: faiss.IndexFlatIP | None = None
# metadata: list of dicts with {"text": str, "page": int, "doc_id": str, "source": str, "kind": "profile"|"chunk"}
metadata: List[dict] = []
# --- NEW: dual indexes for section-aware retrieval ---
sections_index: faiss.IndexFlatIP | None = None
sections_meta: List[dict] = []  # [{"text","doc_id","section_id","section_title","page","kind":"section"}]

chunks_index: faiss.IndexFlatIP | None = None
chunks_meta: List[dict] = []
# docs_registry entries now include optional "profile"
docs_registry: List[dict] = []  # [{doc_id, filename, path, sha256, pages, chunk_count, uploaded_at, profile?}]

# =========================
# Utilities
# =========================
def load_embedder() -> SentenceTransformer:
    global emb_model
    if emb_model is None:
        emb_model = SentenceTransformer(EMBED_MODEL_NAME)
    return emb_model

def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Chunker that prefers sentence/newline breaks near edges."""
    text = re.sub(r"\s+\n", "\n", text).strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        window = text[start:end]
        dot = window.rfind(". ")
        nl = window.rfind("\n")
        cut = max(dot, nl)
        if cut == -1 or end == len(text) or (end - start) < int(chunk_size * 0.6):
            cut = len(window)
        chunk = window[:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, start + chunk_size - overlap)
    return chunks

def pdf_to_chunks(file_path: str) -> List[Tuple[str, int]]:
    """Extract text from a PDF into (chunk, page_number) pairs."""
    pairs: List[Tuple[str, int]] = []
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page_idx, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            txt = txt.strip()
            if not txt:
                continue
            for ch in chunk_text(txt):
                pairs.append((ch, page_idx + 1))
    return pairs

# --- NEW: light-weight section detection ---
SECTION_KEYWORDS = [
    "abstract","introduction","background","related work","method","methods",
    "methodology","approach","experiments","results","evaluation",
    "discussion","conclusion","limitations","future work","references"
]

def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 4: return False
    lo = s.lower()

    # keyword match
    for kw in SECTION_KEYWORDS:
        if lo.startswith(kw): 
            return True

    # numbered headings: 1., 1.2, 2.3.4 Title
    if re.match(r"^\d+(\.\d+){0,3}\s+[A-Za-z].{2,}$", s):
        return True

    # ALL CAPS short-ish
    if len(s) <= 80 and re.match(r"^[A-Z0-9 \-,:;()]+$", s) and not s.endswith("."):
        # avoid shouting paragraphs by requiring at least one space (two+ words)
        return " " in s

    return False

def extract_pages_text(file_path: str) -> list[str]:
    pages = []
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            try:
                txt = (page.extract_text() or "").strip()
            except Exception:
                txt = ""
            pages.append(txt)
    return pages

def detect_sections_from_pages(pages: list[str], fallback_title_prefix="Page") -> list[dict]:
    """
    Returns: list of {"section_id","section_title","page_start","page_end","text"}
    """
    sections = []
    cur_title = None
    cur_start = 1
    cur_buf = []

    def _flush(end_page):
        nonlocal sections, cur_title, cur_start, cur_buf
        if cur_title is None and cur_buf:
            # fallback: page block
            title = f"{fallback_title_prefix} {cur_start}"
        else:
            title = cur_title or f"{fallback_title_prefix} {cur_start}"
        body = "\n".join(cur_buf).strip()
        if body:
            sid = hashlib.sha1(f"{title}-{cur_start}-{end_page}".encode("utf-8")).hexdigest()[:10]
            sections.append({
                "section_id": sid,
                "section_title": title.strip(),
                "page_start": cur_start,
                "page_end": end_page,
                "text": body
            })
        cur_title, cur_buf[:] = None, []

    for i, raw in enumerate(pages, start=1):
        if not raw:
            continue
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        j = 0
        while j < len(lines):
            ln = lines[j]
            if _looks_like_heading(ln):
                # heading encountered: flush previous section
                if cur_buf:
                    _flush(i)
                cur_title = ln
                cur_start = i
                # consume subsequent blank/very short lines as part of title "block"
                j += 1
                # rest of lines for this page belong to this section
                while j < len(lines):
                    cur_buf.append(lines[j])
                    j += 1
            else:
                cur_buf.append(ln)
                j += 1

    # end flush
    if cur_buf:
        _flush(len(pages))

    # if nothing detected at all, make 1 section per page
    if not sections:
        for i, txt in enumerate(pages, start=1):
            if not txt: continue
            sid = hashlib.sha1(f"{i}".encode("utf-8")).hexdigest()[:10]
            sections.append({
                "section_id": sid,
                "section_title": f"{fallback_title_prefix} {i}",
                "page_start": i, "page_end": i,
                "text": txt.strip()
            })
    return sections


def make_doc_profile(pairs: List[Tuple[str, int]], filename: str) -> str:
    """Heuristic profile: title + abstract/intro slice from first ~2 pages."""
    pages = {}
    for t, p in pairs:
        pages.setdefault(p, []).append(t)
    first_two_text = "\n".join(["\n".join(pages.get(i, [])) for i in sorted(pages)[:2]])
    # naive title = first non-empty line
    title = ""
    for line in first_two_text.splitlines():
        s = line.strip()
        if len(s) > 5:
            title = s
            break
    if not title:
        title = filename
    lower = first_two_text.lower()
    abstract = ""
    if "abstract" in lower:
        start = lower.find("abstract")
        abstract = first_two_text[start:start+1200]
    elif "introduction" in lower:
        start = lower.find("introduction")
        abstract = first_two_text[start:start+1200]
    else:
        abstract = first_two_text[:1200]
    profile = f"TITLE: {title}\nSUMMARY: {abstract.strip()}"
    return profile

def save_store(index: faiss.IndexFlatIP, meta: List[dict]) -> None:
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

def save_dual_stores(sec_index, sec_meta, ch_index, ch_meta) -> None:
    faiss.write_index(sec_index, SECT_INDEX_PATH)
    with open(SECT_META_PATH, "wb") as f: pickle.dump(sec_meta, f)
    faiss.write_index(ch_index, CHUNK_INDEX_PATH)
    with open(CHUNK_META_PATH, "wb") as f: pickle.dump(ch_meta, f)

def load_store():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        return None, None
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def load_dual_stores():
    if not (os.path.exists(SECT_INDEX_PATH) and os.path.exists(SECT_META_PATH)
            and os.path.exists(CHUNK_INDEX_PATH) and os.path.exists(CHUNK_META_PATH)):
        return None
    sidx = faiss.read_index(SECT_INDEX_PATH)
    with open(SECT_META_PATH, "rb") as f: smeta = pickle.load(f)
    cidx = faiss.read_index(CHUNK_INDEX_PATH)
    with open(CHUNK_META_PATH, "rb") as f: cmeta = pickle.load(f)
    return sidx, smeta, cidx, cmeta

def load_docs() -> list[dict]:
    if not os.path.exists(DOCS_PATH):
        return []
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_docs(docs: list[dict]) -> None:
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

# =========================
# Retrieval with diversity
# =========================
def retrieve(query: str, k: int = TOP_K, allowed_doc_ids: List[str] | None = None) -> List[dict]:
    """
    Greedy selection with:
      - doc_id+page de-dupe (avoid N chunks from the same page)
      - similarity de-dupe (avoid near-identical chunks across pages/docs)
    """
    assert faiss_index is not None and metadata, "Index not ready."
    model = load_embedder()

    q = model.encode([query], convert_to_numpy=True)
    q = normalize(q.astype(np.float32))

    over_k = max(k * OVERFETCH_MULT, k + 10)
    D, I = faiss_index.search(q, over_k)

    allowed_set = set(allowed_doc_ids) if allowed_doc_ids else None
    cand_meta, cand_texts = [], []
    for idx in I[0]:
        if 0 <= idx < len(metadata):
            m = metadata[idx]
            if allowed_set and m.get("doc_id") not in allowed_set:
                continue
            cand_meta.append(m)
            cand_texts.append(m["text"])

    if not cand_meta:
        return []

    cand_embs = model.encode(cand_texts, convert_to_numpy=True)
    cand_embs = normalize(cand_embs.astype(np.float32))

    results: List[dict] = []
    kept_embs: List[np.ndarray] = []
    seen_pages = set()  # (doc_id, page)

    for j, m in enumerate(cand_meta):
        key = (m.get("doc_id"), m.get("page"))
        if key in seen_pages:
            continue
        e = cand_embs[j]
        if any(float(np.dot(e, ke)) >= DIVERSITY_SIM_THRESHOLD for ke in kept_embs):
            continue
        results.append(m)
        kept_embs.append(e)
        seen_pages.add(key)
        if len(results) >= k:
            break

    if len(results) < k:
        for m in cand_meta:
            key = (m.get("doc_id"), m.get("page"))
            if key in seen_pages:
                continue
            results.append(m)
            seen_pages.add(key)
            if len(results) >= k:
                break
    return results

def retrieve_hierarchical(query: str, allowed_doc_ids: List[str] | None = None) -> List[dict]:
    """
    Stage A: retrieve sections → diversity → select
    Stage B: retrieve chunks globally, then filter to chosen sections → diversity/page de-dupe → final TOP-K
    """
    if sections_index is None or chunks_index is None:
        return []  # fall back to flat retrieve() upstream

    model = load_embedder()
    q = model.encode([query], convert_to_numpy=True)
    q = normalize(q.astype(np.float32))

    # --- Stage A: sections ---
    over_k = max(SECTION_OVERFETCH, SECTIONS_FINAL * OVERFETCH_MULT)
    D_sec, I_sec = sections_index.search(q, over_k)
    cand_secs = []
    allowed = set(allowed_doc_ids) if allowed_doc_ids else None
    for idx in I_sec[0]:
        if 0 <= idx < len(sections_meta):
            m = sections_meta[idx]
            if allowed and m["doc_id"] not in allowed:
                continue
            cand_secs.append(m)
    if not cand_secs:
        return []

    # diversity on sections: avoid duplicate section titles/pages within same doc
    keep_secs, seen = [], set()
    for m in cand_secs:
        key = (m["doc_id"], m["section_title"])
        if key in seen: 
            continue
        keep_secs.append(m); seen.add(key)
        if len(keep_secs) >= SECTIONS_FINAL:
            break

    chosen_section_ids = {m["section_id"] for m in keep_secs}

    # --- Stage B: chunks ---
    # Global overfetch, then filter to chosen sections
    over_chunks = max(GLOBAL_TOP_K * OVERFETCH_MULT, GLOBAL_TOP_K + 20)
    D_ch, I_ch = chunks_index.search(q, over_chunks)

    # gather candidate chunks from selected sections
    cand = []
    for idx in I_ch[0]:
        if 0 <= idx < len(chunks_meta):
            cm = chunks_meta[idx]
            if allowed and cm["doc_id"] not in allowed:
                continue
            if cm.get("section_id") in chosen_section_ids:
                cand.append(cm)

    if not cand:
        # fallback: try flat global top-k filtered by allowed docs
        return retrieve(query, k=GLOBAL_TOP_K, allowed_doc_ids=allowed_doc_ids)

    # diversity: avoid same (doc_id, page), avoid near-duplicate
    results, kept_texts, seen_pages = [], [], set()
    for m in cand:
        key = (m["doc_id"], m.get("page", -1))
        if key in seen_pages:
            continue
        # simple similarity de-dupe using text hashes (cheaper than re-embeddings)
        txt_sig = hashlib.sha1(m["text"][:400].encode("utf-8")).hexdigest()[:12]
        if txt_sig in kept_texts:
            continue
        results.append(m)
        kept_texts.append(txt_sig)
        seen_pages.add(key)
        if len(results) >= GLOBAL_TOP_K:
            break

    # If still thin, top up with remaining chunk hits (relaxed)
    if len(results) < GLOBAL_TOP_K:
        for m in cand:
            key = (m["doc_id"], m.get("page", -1))
            if key in seen_pages:
                continue
            results.append(m)
            seen_pages.add(key)
            if len(results) >= GLOBAL_TOP_K:
                break

    return results


# =========================
# Prompt budgeting helpers
# =========================
def _smart_cut(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    window = text[:max_len]
    nl = window.rfind("\n"); dot = window.rfind(". ")
    cut = max(nl, dot)
    if cut == -1 or cut < int(max_len * 0.6):
        cut = max_len
    return window[:cut].strip()

def _budget_chunks(chunks: List[str], question: str, limit: int) -> List[str]:
    instr = "Use ONLY the context to answer the question. If the answer is not in the context, say you don't know.\n"
    sep_overhead = 10 * max(0, len(chunks) - 1)
    overhead = len(instr) + len("Context:\n") + len("\nQuestion:\n") + len("\nHelpful Answer:") + sep_overhead + len(question)
    budget = max(0, limit - overhead)
    if budget <= 0:
        return [_smart_cut(chunks[0], max(128, limit // 2))]
    shares = [budget // len(chunks)] * len(chunks)
    for i in range(budget - sum(shares)):
        shares[i] += 1
    out, carry = [], 0
    for i, ch in enumerate(chunks):
        share = shares[i] + carry
        if len(ch) <= share:
            out.append(ch); carry = share - len(ch)
        else:
            out.append(_smart_cut(ch, max(0, share))); carry = 0
    return out

def render_prompt(context_texts: List[str], question: str) -> str:
    packed = _budget_chunks(context_texts, question, CONTEXT_CHAR_LIMIT)
    ctx = "\n\n".join(packed)
    return (
        "Use ONLY the context to answer the question. If the answer is not in the context, say you don't know.\n"
        f"Context:\n{ctx}\n"
        f"Question: {question}\n"
        "Helpful Answer:"
    )

# =========================
# Ollama call + postprocess
# =========================
def ollama_generate(prompt: str, max_tokens: int = MAX_NEW_TOKENS, temperature: float = TEMPERATURE) -> str:
    url = f"{MODEL_BASE_URL}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": int(max_tokens),
            "temperature": float(temperature),
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        parsed = json.loads(body)
    return parsed.get("response", "")

def postprocess_answer(text: str) -> str:
    out = text or ""
    if "Helpful Answer:" in out:
        out = out.split("Helpful Answer:", 1)[1].strip()
    for stop in ("\nContext:", "\nQuestion:", "\nHelpful Answer:"):
        i = out.find(stop)
        if i != -1:
            out = out[:i].strip()
    return out.strip()

def _file_sha256(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

# =========================
# API models
# =========================
class ChatRequest(BaseModel):
    message: str
    doc_ids: List[str] | None = None  # optional filter

class IndexRequest(BaseModel):
    doc_ids: List[str] | None = None  # if None, index all

# =========================
# Endpoints
# =========================
@app.post("/upload")
async def upload_pdf(file: UploadFile):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Please upload a PDF.")
    raw = await file.read()
    file_hash = _file_sha256(raw)

    # dedupe on hash
    existing = next((d for d in docs_registry if d.get("sha256") == file_hash), None)
    if existing:
        return {
            "status": "exists",
            "filename": existing["filename"],
            "doc_id": existing["doc_id"],
            "pages": existing.get("pages", 0),
            "chunk_count": existing.get("chunk_count", 0),
        }

    # persist file
    doc_id = uuid.uuid4().hex[:8]
    out_name = f"{doc_id}_{file.filename}"
    out_path = Path(UPLOAD_DIR) / out_name
    with open(out_path, "wb") as f:
        f.write(raw)

    record = {
        "doc_id": doc_id,
        "filename": file.filename,
        "path": str(out_path),
        "sha256": file_hash,
        "pages": 0,
        "chunk_count": 0,
        "uploaded_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        # "profile": will be populated at index time
    }
    docs_registry.append(record)
    save_docs(docs_registry)
    return {"status": "staged", "filename": file.filename, "doc_id": doc_id}

@app.post("/index")
def build_index(req: IndexRequest):
    global faiss_index, metadata, docs_registry
    chosen = [d for d in docs_registry if (not req.doc_ids or d["doc_id"] in req.doc_ids)]
    if not chosen:
        raise HTTPException(status_code=400, detail="No documents selected to index.")

    model = load_embedder()

    all_sec_texts, all_sec_meta = [], []
    all_chunk_texts, all_chunk_meta = [], []

    for d in chosen:
        pages = extract_pages_text(d["path"])
        secs = detect_sections_from_pages(pages, fallback_title_prefix=d["filename"])

        # per-doc stats
        d["pages"] = sum(1 for p in pages if p)
        d["section_count"] = len(secs)
        d["chunk_count"] = 0  # will update below

        # profile (keep your existing behavior)
        pairs_for_profile = []
        # reuse your make_doc_profile(): build (text, page) pairs from first 2 pages
        # We'll reconstruct minimal pairs to feed make_doc_profile()
        for idx, txt in enumerate(pages[:2], start=1):
            if txt:
                for ch in chunk_text(txt):
                    pairs_for_profile.append((ch, idx))
        profile_text = make_doc_profile(pairs_for_profile, d["filename"])
        d["profile"] = profile_text
        # Option: store profile as a special "section" so it can be retrieved in Stage A
        all_sec_texts.append(profile_text)
        all_sec_meta.append({
            "text": profile_text, "page": 0, "doc_id": d["doc_id"],
            "source": d["filename"], "kind": "section",
            "section_id": f"__PROFILE__-{d['doc_id']}",
            "section_title": "__PROFILE__"
        })

        # sections + chunks
        chunk_counter = 0
        for s in secs:
            # section record (title + leading snippet makes it more representative)
            sec_repr = (s["section_title"] + "\n" + s["text"][:1000]).strip()
            all_sec_texts.append(sec_repr)
            all_sec_meta.append({
                "text": sec_repr,
                "page": s["page_start"],
                "doc_id": d["doc_id"],
                "source": d["filename"],
                "kind": "section",
                "section_id": s["section_id"],
                "section_title": s["section_title"],
            })

            # chunk within the section
            for ch in chunk_text(s["text"]):
                all_chunk_texts.append(ch)
                all_chunk_meta.append({
                    "text": ch,
                    "page": s["page_start"],
                    "doc_id": d["doc_id"],
                    "source": d["filename"],
                    "kind": "chunk",
                    "section_id": s["section_id"],
                    "section_title": s["section_title"],
                })
                chunk_counter += 1

        d["chunk_count"] = chunk_counter

    # build + persist dual indexes
    if not all_sec_texts or not all_chunk_texts:
        raise HTTPException(status_code=400, detail="No text extracted from selected documents.")

    # sections
    sec_embs = model.encode(all_sec_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    sec_embs = normalize(sec_embs.astype(np.float32))
    s_dim = sec_embs.shape[1]
    s_idx = faiss.IndexFlatIP(s_dim)
    s_idx.add(sec_embs)

    # chunks
    chk_embs = model.encode(all_chunk_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    chk_embs = normalize(chk_embs.astype(np.float32))
    c_dim = chk_embs.shape[1]
    c_idx = faiss.IndexFlatIP(c_dim)
    c_idx.add(chk_embs)

    # swap in-memory + save
    global sections_index, sections_meta, chunks_index, chunks_meta, faiss_index, metadata
    sections_index, sections_meta = s_idx, all_sec_meta
    chunks_index, chunks_meta = c_idx, all_chunk_meta

    # keep your old flat index for fallback (optional)
    faiss_index = c_idx
    metadata = all_chunk_meta

    save_dual_stores(sections_index, sections_meta, chunks_index, chunks_meta)
    save_docs(docs_registry)

    return {"status": "indexed", "docs_indexed": len(chosen),
            "sections": len(all_sec_meta), "total_chunks": len(all_chunk_meta)}


@app.delete("/docs/{doc_id}")
def delete_doc(doc_id: str):
    global docs_registry
    before = len(docs_registry)
    docs_registry = [d for d in docs_registry if d["doc_id"] != doc_id]
    save_docs(docs_registry)
    return {"status": "deleted", "removed": before - len(docs_registry)}

@app.post("/chat")
async def chat(req: ChatRequest):
    global faiss_index, metadata, docs_registry
    if faiss_index is None or not metadata:
        raise HTTPException(status_code=503, detail="Index not ready. Add files and click 'Build Index'.")

    question = (req.message or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message is empty.")

    try:
        allowed = req.doc_ids if req.doc_ids else None

        # retrieve passages
        hits = retrieve_hierarchical(question, allowed_doc_ids=allowed)
        if not hits:
            # fallback to flat if section-aware has no signal
            hits = retrieve(question, k=TOP_K, allowed_doc_ids=allowed)
        
        # profile-first logic
        should_add_profile = False
        q_low = question.lower()
        if allowed and len(allowed) == 1:
            should_add_profile = True
        elif any(t in q_low for t in OVERVIEW_TRIGGERS):
            should_add_profile = True

        if should_add_profile:
            # choose which doc's profile to include
            if allowed and len(allowed) == 1:
                target_doc = allowed[0]
            else:
                # doc with most hits
                doc_counts = {}
                for h in hits:
                    doc_counts[h["doc_id"]] = doc_counts.get(h["doc_id"], 0) + 1
                target_doc = max(doc_counts, key=doc_counts.get)

            prof_text = None
            prof_filename = None
            for d in docs_registry:
                if d["doc_id"] == target_doc and d.get("profile"):
                    prof_text = d["profile"]
                    prof_filename = d["filename"]
                    break

            if prof_text:
                # prepend profile if not already in hits
                if not any(h.get("kind") == "profile" and h["doc_id"] == target_doc for h in hits):
                    hits = [{
                        "text": prof_text, "page": 0, "doc_id": target_doc,
                        "source": prof_filename or "profile", "kind": "profile"
                    }] + hits

        # debug
        print("RETRIEVED:", [
            (h.get("source"), h.get("page"), h.get("kind", "chunk"),
             (h["text"][:80] + "...") if len(h["text"]) > 80 else h["text"])
            for h in hits
        ])

        # prompt
        context_texts = [h["text"] for h in hits]
        prompt = render_prompt(context_texts, question)

        # generation (slightly larger budget for overviews)
        max_tokens = MAX_NEW_TOKENS
        if should_add_profile:
            max_tokens = max(max_tokens, 220)

        raw = ollama_generate(prompt, max_tokens=max_tokens, temperature=TEMPERATURE)
        answer = postprocess_answer(raw)
        return {
            "response": answer,
            "sources": [
                {"doc_id": h["doc_id"], "page": h["page"], "source": h["source"]}
                for h in hits
            ]
        }

    except urllib.error.URLError as e:
        raise HTTPException(status_code=503, detail=f"Ollama not reachable at {MODEL_BASE_URL}: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chat: {e}")

@app.get("/docs")
def list_docs():
    return {"docs": docs_registry, "total_docs": len(docs_registry)}

# =========================
# Startup: load persisted index + registry
# =========================
@app.on_event("startup")
def _try_load_store():
    global faiss_index, metadata, docs_registry
    idx, meta = load_store()
    dual = load_dual_stores()
    if dual is not None:
        global sections_index, sections_meta, chunks_index, chunks_meta
        sections_index, sections_meta, chunks_index, chunks_meta = dual
    if idx is not None and meta is not None:
        faiss_index, metadata = idx, meta
    docs_registry = load_docs()
