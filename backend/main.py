import os
import re
import json
import pickle
import uuid
import hashlib
import logging
import shutil
import time
import urllib.request
import urllib.error
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
import sys
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
from fastapi import Depends, FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session

if __package__:
    try:
        from .auth import (
            create_access_token,
            get_current_user,
            hash_password,
            verify_password,
        )
        from .db import SessionLocal, get_db, init_db
        from .models import ChatMessage, ChatSession, Chunk, Document, Section, User
    except ModuleNotFoundError as exc:
        if exc.name == "jwt":
            raise ModuleNotFoundError("Missing dependency \"PyJWT\". Run `pip install -r requirements.txt` in the backend virtualenv.") from exc
        raise
else:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    try:
        from auth import (
            create_access_token,
            get_current_user,
            hash_password,
            verify_password,
        )
        from db import SessionLocal, get_db, init_db
        from models import ChatMessage, ChatSession, Chunk, Document, Section, User
    except ModuleNotFoundError as exc:
        if exc.name == "jwt":
            raise ModuleNotFoundError("Missing dependency \"PyJWT\". Run `pip install -r requirements.txt` in the backend virtualenv.") from exc
        raise
# =========================
# Config (env-tunable)
# =========================
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3:8b")

TOP_K = int(os.getenv("TOP_K", "4"))                      # retrieval depth
CONTEXT_CHAR_LIMIT = int(os.getenv("CONTEXT_CHAR_LIMIT", "2500"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "150"))  # default answer length
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploaded_docs"))
STORE_DIR  = os.getenv("STORE_DIR",  str(BASE_DIR / "store"))
LEGACY_INDEX_PATH = Path(STORE_DIR) / "index.faiss"
# --- NEW: dual-index paths for section-aware retrieval ---
LEGACY_SECT_INDEX_PATH = Path(STORE_DIR) / "sections.index.faiss"
LEGACY_SECT_META_PATH = Path(STORE_DIR) / "sections.meta.pkl"
LEGACY_CHUNK_INDEX_PATH = Path(STORE_DIR) / "chunks.index.faiss"
LEGACY_CHUNK_META_PATH = Path(STORE_DIR) / "chunks.meta.pkl"

# Section-aware retrieval knobs
SECTION_OVERFETCH = int(os.getenv("SECTION_OVERFETCH", "12"))
SECTIONS_FINAL = int(os.getenv("SECTIONS_FINAL", "6"))        # keep this many sections after diversity
CHUNKS_PER_SECTION = int(os.getenv("CHUNKS_PER_SECTION", "2")) # zoom-in per section
GLOBAL_TOP_K = int(os.getenv("GLOBAL_TOP_K", "6"))             # final chunk count (can be >= TOP_K)

LEGACY_META_PATH = Path(STORE_DIR) / "meta.pkl"
LEGACY_DOCS_PATH = Path(STORE_DIR) / "docs.json"

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DIVERSITY_SIM_THRESHOLD = float(os.getenv("DIVERSITY_SIM_THRESHOLD", "0.95"))
OVERFETCH_MULT = int(os.getenv("OVERFETCH_MULT", "5"))

# Heuristic triggers for overview questions
OVERVIEW_TRIGGERS = tuple(
    s.strip() for s in os.getenv(
        "OVERVIEW_TRIGGERS",
        "what is;whatâ€™s;whats;overview;summary;high level;broadly;explain the paper;describe the paper"
    ).split(";")
    if s.strip()
)

_NUMERIC_LOG_LEVEL = getattr(logging, LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=_NUMERIC_LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("rag_backend")
logger.setLevel(_NUMERIC_LOG_LEVEL)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STORE_DIR, exist_ok=True)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    logger.info(
        "Starting FastAPI application",
    )
    logger.info(
        "Config: model=%s embed_model=%s top_k=%s global_top_k=%s",
        MODEL_NAME,
        EMBED_MODEL_NAME,
        TOP_K,
        GLOBAL_TOP_K,
    )
    try:
        init_db()
        with SessionLocal() as db:
            migrate_legacy_docs(db)
            doc_total = db.scalar(select(func.count()).select_from(Document)) or 0
            user_total = db.scalar(select(func.count()).select_from(User)) or 0
            logger.info("Database ready (users=%d docs=%d)", user_total, doc_total)
    except Exception:
        logger.exception("Failed to initialise database")
    yield
    logger.info("FastAPI application shutdown complete")

# =========================
# FastAPI app
# =========================
app = FastAPI(docs_url="/api", redoc_url=None, openapi_url="/openapi.json", lifespan=app_lifespan)
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


@dataclass
class StoreState:
    faiss_index: Optional[faiss.IndexFlatIP] = None
    metadata: Optional[List[dict]] = None
    sections_index: Optional[faiss.IndexFlatIP] = None
    sections_meta: Optional[List[dict]] = None
    chunks_index: Optional[faiss.IndexFlatIP] = None
    chunks_meta: Optional[List[dict]] = None


store_cache: Dict[int, StoreState] = {}
store_cache_lock = Lock()

# =========================
# Utilities
# =========================
def load_embedder() -> SentenceTransformer:
    global emb_model
    if emb_model is None:
        logger.info("Loading embedding model %s", EMBED_MODEL_NAME)
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

def _parse_uploaded_at(value: str | None) -> datetime:
    if not value:
        return datetime.utcnow()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.utcnow()


def migrate_legacy_docs(db: Session) -> None:
    if not LEGACY_DOCS_PATH.exists():
        return
    existing_docs = db.scalar(select(func.count()).select_from(Document)) or 0
    if existing_docs:
        return
    try:
        with LEGACY_DOCS_PATH.open("r", encoding="utf-8") as f:
            legacy_docs = json.load(f)
    except Exception as exc:  # pragma: no cover - legacy path only
        logger.warning("Failed to read legacy docs.json: %s", exc)
        return

    imported = 0
    for entry in legacy_docs:
        doc_id = entry.get("doc_id") or uuid.uuid4().hex[:8]
        if db.execute(select(Document).where(Document.doc_id == doc_id)).scalar_one_or_none():
            continue
        document = Document(
            doc_id=doc_id,
            owner_id=None,
            filename=entry.get("filename", ""),
            path=entry.get("path", ""),
            sha256=entry.get("sha256", ""),
            pages=entry.get("pages", 0) or 0,
            section_count=entry.get("section_count", 0) or entry.get("sections", 0) or 0,
            chunk_count=entry.get("chunk_count", 0) or 0,
            uploaded_at=_parse_uploaded_at(entry.get("uploaded_at")),
            profile=entry.get("profile"),
            shared=True,
        )
        db.add(document)
        imported += 1
    if imported:
        db.commit()
        backup_path = LEGACY_DOCS_PATH.with_suffix(".migrated.json")
        try:
            LEGACY_DOCS_PATH.rename(backup_path)
        except OSError:
            logger.warning("Could not rename legacy docs file after migration")
        logger.info("Migrated %d legacy documents into database", imported)


def user_store_dir(user_id: int) -> Path:
    base = Path(STORE_DIR) / f"user_{user_id}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def store_paths(user_id: int) -> dict[str, Path]:
    base = user_store_dir(user_id)
    return {
        "index": base / "index.faiss",
        "meta": base / "meta.pkl",
        "sections_index": base / "sections.index.faiss",
        "sections_meta": base / "sections.meta.pkl",
        "chunks_index": base / "chunks.index.faiss",
        "chunks_meta": base / "chunks.meta.pkl",
    }


def clear_user_store(user_id: int) -> None:
    with store_cache_lock:
        store_cache.pop(user_id, None)
    for path in store_paths(user_id).values():
        try:
            if path.exists():
                path.unlink()
        except OSError:
            logger.warning("Failed to remove store file %s", path)


def load_user_store_from_disk(user_id: int) -> StoreState | None:
    paths = store_paths(user_id)
    try:
        if not (paths["index"].exists() and paths["meta"].exists()):
            return None
        idx = faiss.read_index(str(paths["index"]))
        with paths["meta"].open("rb") as f:
            meta = pickle.load(f)

        sec_idx = None
        sec_meta = None
        ch_idx = None
        ch_meta = None
        if (
            paths["sections_index"].exists()
            and paths["sections_meta"].exists()
            and paths["chunks_index"].exists()
            and paths["chunks_meta"].exists()
        ):
            sec_idx = faiss.read_index(str(paths["sections_index"]))
            with paths["sections_meta"].open("rb") as f:
                sec_meta = pickle.load(f)
            ch_idx = faiss.read_index(str(paths["chunks_index"]))
            with paths["chunks_meta"].open("rb") as f:
                ch_meta = pickle.load(f)
        return StoreState(
            faiss_index=idx,
            metadata=meta,
            sections_index=sec_idx,
            sections_meta=sec_meta,
            chunks_index=ch_idx,
            chunks_meta=ch_meta,
        )
    except Exception:
        logger.exception("Failed to load user %s store from disk", user_id)
        return None


def persist_user_store(user_id: int, state: StoreState) -> None:
    paths = store_paths(user_id)
    if state.faiss_index is not None and state.metadata is not None:
        faiss.write_index(state.faiss_index, str(paths["index"]))
        with paths["meta"].open("wb") as f:
            pickle.dump(state.metadata, f)
    if (
        state.sections_index is not None
        and state.sections_meta is not None
        and state.chunks_index is not None
        and state.chunks_meta is not None
    ):
        faiss.write_index(state.sections_index, str(paths["sections_index"]))
        with paths["sections_meta"].open("wb") as f:
            pickle.dump(state.sections_meta, f)
        faiss.write_index(state.chunks_index, str(paths["chunks_index"]))
        with paths["chunks_meta"].open("wb") as f:
            pickle.dump(state.chunks_meta, f)


def get_store_state(user_id: int) -> StoreState | None:
    with store_cache_lock:
        cached = store_cache.get(user_id)
    if cached and cached.faiss_index is not None and cached.metadata:
        return cached
    loaded = load_user_store_from_disk(user_id)
    if loaded:
        with store_cache_lock:
            store_cache[user_id] = loaded
    return loaded

# =========================
# Retrieval with diversity
# =========================
def retrieve(store: StoreState, query: str, k: int = TOP_K, allowed_doc_ids: List[str] | None = None) -> List[dict]:
    """
    Greedy selection with:
      - doc_id+page de-dupe (avoid N chunks from the same page)
      - similarity de-dupe (avoid near-identical chunks across pages/docs)
    """
    if store.faiss_index is None or not store.metadata:
        raise HTTPException(status_code=503, detail="Index not ready. Build the index first.")
    model = load_embedder()

    q = model.encode([query], convert_to_numpy=True)
    q = normalize(q.astype(np.float32))

    over_k = max(k * OVERFETCH_MULT, k + 10)
    D, I = store.faiss_index.search(q, over_k)

    allowed_set = set(allowed_doc_ids) if allowed_doc_ids else None
    cand_meta, cand_texts = [], []
    for idx in I[0]:
        if 0 <= idx < len(store.metadata):
            meta_entry = store.metadata[idx]
            if allowed_set and meta_entry.get("doc_id") not in allowed_set:
                continue
            cand_meta.append(meta_entry)
            cand_texts.append(meta_entry["text"])

    if not cand_meta:
        return []

    cand_embs = model.encode(cand_texts, convert_to_numpy=True)
    cand_embs = normalize(cand_embs.astype(np.float32))

    results: List[dict] = []
    kept_embs: List[np.ndarray] = []
    seen_pages = set()  # (doc_id, page)

    for j, meta_entry in enumerate(cand_meta):
        key = (meta_entry.get("doc_id"), meta_entry.get("page"))
        if key in seen_pages:
            continue
        embedding = cand_embs[j]
        if any(float(np.dot(embedding, kept)) >= DIVERSITY_SIM_THRESHOLD for kept in kept_embs):
            continue
        results.append(meta_entry)
        kept_embs.append(embedding)
        seen_pages.add(key)
        if len(results) >= k:
            break

    if len(results) < k:
        for meta_entry in cand_meta:
            key = (meta_entry.get("doc_id"), meta_entry.get("page"))
            if key in seen_pages:
                continue
            results.append(meta_entry)
            seen_pages.add(key)
            if len(results) >= k:
                break
    return results


def retrieve_hierarchical(store: StoreState, query: str, allowed_doc_ids: List[str] | None = None) -> List[dict]:
    """
    Stage A: retrieve sections + diversity + select
    Stage B: retrieve chunks globally, then filter to chosen sections + diversity/page de-dupe + final TOP-K
    """
    if store.sections_index is None or store.chunks_index is None:
        return []

    model = load_embedder()
    q = model.encode([query], convert_to_numpy=True)
    q = normalize(q.astype(np.float32))

    over_k = max(SECTION_OVERFETCH, SECTIONS_FINAL * OVERFETCH_MULT)
    D_sec, I_sec = store.sections_index.search(q, over_k)
    cand_secs = []
    allowed = set(allowed_doc_ids) if allowed_doc_ids else None
    sections_meta = store.sections_meta or []
    for idx in I_sec[0]:
        if 0 <= idx < len(sections_meta):
            meta_entry = sections_meta[idx]
            if allowed and meta_entry["doc_id"] not in allowed:
                continue
            cand_secs.append(meta_entry)
    if not cand_secs:
        return []

    keep_secs, seen = [], set()
    for meta_entry in cand_secs:
        key = (meta_entry["doc_id"], meta_entry.get("section_id"))
        if key in seen:
            continue
        keep_secs.append(meta_entry)
        seen.add(key)
        if len(keep_secs) >= SECTIONS_FINAL:
            break

    chosen_section_ids = {meta_entry["section_id"] for meta_entry in keep_secs}
    chosen_docs = {meta_entry["doc_id"] for meta_entry in keep_secs}

    over_chunks = max(GLOBAL_TOP_K * OVERFETCH_MULT, GLOBAL_TOP_K + 20)
    D_chunks, I_chunks = store.chunks_index.search(q, over_chunks)
    chunks_meta = store.chunks_meta or []
    candidates = []
    for idx in I_chunks[0]:
        if 0 <= idx < len(chunks_meta):
            chunk_entry = chunks_meta[idx]
            if allowed and chunk_entry["doc_id"] not in allowed:
                continue
            if chunk_entry["doc_id"] not in chosen_docs:
                continue
            if chunk_entry.get("section_id") in chosen_section_ids:
                candidates.append(chunk_entry)

    if not candidates:
        return retrieve(store, query, k=GLOBAL_TOP_K, allowed_doc_ids=allowed_doc_ids)

    results: List[dict] = []
    seen_pages = set()
    seen_chunks: set[str] = set()
    for entry in candidates:
        key = (entry.get("doc_id"), entry.get("page"))
        if key in seen_pages:
            continue
        chunk_key = f"{entry.get('doc_id')}:{entry.get('section_id')}:{entry.get('page')}"
        if chunk_key in seen_chunks:
            continue
        results.append(entry)
        seen_pages.add(key)
        seen_chunks.add(chunk_key)
        if len(results) >= GLOBAL_TOP_K:
            break

    return results


def serialize_document(document: Document) -> dict:
    return {
        "doc_id": document.doc_id,
        "filename": document.filename,
        "pages": document.pages,
        "section_count": document.section_count,
        "chunk_count": document.chunk_count,
        "uploaded_at": (document.uploaded_at or datetime.utcnow()).isoformat() + 'Z',
        "profile": document.profile,
        "shared": document.shared,
    }


def accessible_documents_query(user: User, include_shared: bool = True, doc_ids: Optional[List[str]] = None):
    stmt = select(Document)
    filters = []
    if include_shared:
        filters.append(Document.shared.is_(True))
        filters.append(Document.owner_id.is_(None))
    filters.append(Document.owner_id == user.id)
    stmt = stmt.where(or_(*filters))
    if doc_ids:
        stmt = stmt.where(Document.doc_id.in_(doc_ids))
    return stmt


def get_accessible_documents(db: Session, user: User, doc_ids: Optional[List[str]] = None) -> List[Document]:
    stmt = accessible_documents_query(user, doc_ids=doc_ids)
    docs = db.execute(stmt).scalars().all()
    if doc_ids and {d.doc_id for d in docs} != set(doc_ids):
        raise HTTPException(status_code=403, detail="One or more documents are not accessible.")
    return docs


def ensure_store_for_user(user_id: int) -> StoreState:
    store = get_store_state(user_id)
    if store is None or store.faiss_index is None or not store.metadata:
        raise HTTPException(status_code=503, detail="Index not ready. Build the index first.")
    return store


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
    last_error: urllib.error.URLError | None = None
    for attempt in range(OLLAMA_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            response = parsed.get("response", "")
            logger.debug(
                "Ollama generation succeeded (attempt %d, chars=%d)",
                attempt + 1,
                len(response),
            )
            return response
        except urllib.error.HTTPError as http_err:
            logger.error(
                "Ollama HTTP error on attempt %d/%d: %s", 
                attempt + 1,
                OLLAMA_RETRIES + 1,
                http_err,
            )
            raise
        except urllib.error.URLError as url_err:
            last_error = url_err
            logger.warning(
                "Ollama connection failed (attempt %d/%d): %s",
                attempt + 1,
                OLLAMA_RETRIES + 1,
                url_err,
            )
            if attempt < OLLAMA_RETRIES:
                sleep_for = min(2 ** attempt, 5.0)
                time.sleep(sleep_for)
        except json.JSONDecodeError as decode_err:
            logger.error("Failed to decode Ollama response: %s", decode_err)
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("Ollama did not return a response")

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
    session_id: Optional[str] = None


class IndexRequest(BaseModel):
    doc_ids: List[str] | None = None  # if None, index all


class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# =========================
# Endpoints
# =========================
@app.post("/auth/signup", response_model=TokenResponse)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")
    existing = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="Email is already registered.")
    user_count = db.scalar(select(func.count()).select_from(User)) or 0
    role = "admin" if user_count == 0 else "user"
    user = User(email=email, password_hash=hash_password(payload.password), role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    if role == "admin":
        orphan_docs = db.execute(select(Document).where(Document.owner_id.is_(None))).scalars().all()
        changed = False
        for doc in orphan_docs:
            doc.owner_id = user.id
            changed = True
        if changed:
            db.commit()
    token = create_access_token(subject=user.id)
    logger.info("User signed up: %s (role=%s)", email, role)
    return TokenResponse(access_token=token)


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = create_access_token(subject=user.id)
    logger.info("User logged in: %s", email)
    return TokenResponse(access_token=token)

@app.post("/upload")
async def upload_pdf(
    file: UploadFile,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Please upload a PDF.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    file_hash = _file_sha256(raw)
    logger.info(
        "Upload received: filename=%s size=%d bytes",
        file.filename,
        len(raw),
    )

    existing = db.execute(
        select(Document).where(Document.sha256 == file_hash, Document.owner_id == current_user.id)
    ).scalar_one_or_none()
    if existing:
        logger.info(
            "Upload skipped (duplicate for user): filename=%s doc_id=%s",
            file.filename,
            existing.doc_id,
        )
        return {
            "status": "exists",
            "filename": existing.filename,
            "doc_id": existing.doc_id,
            "pages": existing.pages,
            "chunk_count": existing.chunk_count,
        }

    doc_id = uuid.uuid4().hex[:8]
    out_name = f"{doc_id}_{file.filename}"
    out_path = Path(UPLOAD_DIR) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(raw)

    document = Document(
        doc_id=doc_id,
        owner_id=current_user.id,
        filename=file.filename,
        path=str(out_path),
        sha256=file_hash,
        uploaded_at=datetime.utcnow(),
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    clear_user_store(current_user.id)

    logger.info("File stored: filename=%s doc_id=%s user=%s", file.filename, doc_id, current_user.email)
    return {"status": "staged", "filename": document.filename, "doc_id": document.doc_id}

@app.post("/index")
def build_index(
    req: IndexRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc_ids = req.doc_ids if req.doc_ids else None
    documents = get_accessible_documents(db, current_user, doc_ids=doc_ids)
    if not documents:
        raise HTTPException(status_code=400, detail="No documents selected to index.")

    logger.info("Building vector indexes for %d document(s)", len(documents))
    clear_user_store(current_user.id)

    model = load_embedder()

    all_sec_texts: List[str] = []
    all_sec_meta: List[dict] = []
    all_chunk_texts: List[str] = []
    all_chunk_meta: List[dict] = []

    for document in documents:
        path = Path(document.path)
        if not path.exists():
            logger.warning("Document file missing on disk for doc_id=%s", document.doc_id)
            continue

        pages = extract_pages_text(str(path))
        sections = detect_sections_from_pages(pages, fallback_title_prefix=document.filename)

        document.pages = sum(1 for p in pages if p)
        document.section_count = len(sections)

        pairs_for_profile: List[Tuple[str, int]] = []
        for page_idx, txt in enumerate(pages[:2], start=1):
            if not txt:
                continue
            for chunk in chunk_text(txt):
                pairs_for_profile.append((chunk, page_idx))
        profile_text = make_doc_profile(pairs_for_profile, document.filename)
        document.profile = profile_text
        all_sec_texts.append(profile_text)
        all_sec_meta.append({
            "text": profile_text,
            "page": 0,
            "doc_id": document.doc_id,
            "source": document.filename,
            "kind": "section",
            "section_id": f"__PROFILE__-{document.doc_id}",
            "section_title": "__PROFILE__",
        })

        section_rows: List[Section] = []
        chunk_rows: List[Chunk] = []
        chunk_counter = 0

        for section in sections:
            section_repr = (section["section_title"] + "\n" + section["text"][:1000]).strip()
            all_sec_texts.append(section_repr)
            all_sec_meta.append({
                "text": section_repr,
                "page": section["page_start"],
                "doc_id": document.doc_id,
                "source": document.filename,
                "kind": "section",
                "section_id": section["section_id"],
                "section_title": section["section_title"],
            })

            section_rows.append(
                Section(
                    document_id=document.id,
                    section_id=section["section_id"],
                    section_title=section["section_title"],
                    page_start=section["page_start"],
                    page_end=section["page_end"],
                    text=section["text"],
                )
            )

            for chunk in chunk_text(section["text"]):
                all_chunk_texts.append(chunk)
                all_chunk_meta.append({
                    "text": chunk,
                    "page": section["page_start"],
                    "doc_id": document.doc_id,
                    "source": document.filename,
                    "kind": "chunk",
                    "section_id": section["section_id"],
                    "section_title": section["section_title"],
                })
                chunk_rows.append(
                    Chunk(
                        document_id=document.id,
                        section_id=section["section_id"],
                        section_title=section["section_title"],
                        page=section["page_start"],
                        text=chunk,
                    )
                )
                chunk_counter += 1

        document.chunk_count = chunk_counter

        db.execute(delete(Section).where(Section.document_id == document.id))
        db.execute(delete(Chunk).where(Chunk.document_id == document.id))
        for row in section_rows:
            db.add(row)
        for row in chunk_rows:
            db.add(row)

    if not all_sec_texts or not all_chunk_texts:
        db.commit()
        logger.warning("Index build aborted: no text extracted from selected documents")
        raise HTTPException(status_code=400, detail="No text extracted from selected documents.")

    sec_embs = model.encode(all_sec_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    sec_embs = normalize(sec_embs.astype(np.float32))
    sections_index = faiss.IndexFlatIP(sec_embs.shape[1])
    sections_index.add(sec_embs)

    chk_embs = model.encode(all_chunk_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    chk_embs = normalize(chk_embs.astype(np.float32))
    chunks_index = faiss.IndexFlatIP(chk_embs.shape[1])
    chunks_index.add(chk_embs)

    db.commit()

    store_state = StoreState(
        faiss_index=chunks_index,
        metadata=all_chunk_meta,
        sections_index=sections_index,
        sections_meta=all_sec_meta,
        chunks_index=chunks_index,
        chunks_meta=all_chunk_meta,
    )

    with store_cache_lock:
        store_cache[current_user.id] = store_state
    persist_user_store(current_user.id, store_state)

    logger.info(
        "Index build complete: sections=%d chunks=%d",
        len(all_sec_meta),
        len(all_chunk_meta),
    )

    return {
        "status": "indexed",
        "docs_indexed": len(documents),
        "sections": len(all_sec_meta),
        "total_chunks": len(all_chunk_meta),
    }

@app.delete("/docs/{doc_id}")
def delete_doc(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    document = db.execute(select(Document).where(Document.doc_id == doc_id)).scalar_one_or_none()
    if document is None:
        logger.warning("Delete requested for unknown document %s", doc_id)
        return {"status": "deleted", "removed": 0}

    if document.owner_id not in (current_user.id, None) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="You cannot delete this document.")

    db.execute(delete(Section).where(Section.document_id == document.id))
    db.execute(delete(Chunk).where(Chunk.document_id == document.id))
    db.delete(document)
    db.commit()

    try:
        path = Path(document.path)
        if path.exists():
            path.unlink()
    except OSError:
        logger.warning("Failed to remove file for deleted document %s", doc_id)

    clear_user_store(current_user.id)
    logger.info("Deleted document %s for user %s", doc_id, current_user.email)
    return {"status": "deleted", "removed": 1}

@app.post("/chat")
async def chat(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    question = (req.message or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message is empty.")

    docs = get_accessible_documents(db, current_user, doc_ids=req.doc_ids if req.doc_ids else None)
    if not docs:
        raise HTTPException(status_code=400, detail="No accessible documents selected.")
    doc_lookup = {doc.doc_id: doc for doc in docs}

    store = ensure_store_for_user(current_user.id)

    allowed_ids = req.doc_ids if req.doc_ids else list(doc_lookup.keys())
    preview = (question[:80] + ("..." if len(question) > 80 else "")).replace("\n", " ")
    logger.info(
        "Chat request received (docs=%s, preview=%s)",
        allowed_ids if allowed_ids else "all",
        preview,
    )

    try:
        hits = retrieve_hierarchical(store, question, allowed_doc_ids=allowed_ids)
        if not hits:
            hits = retrieve(store, question, k=TOP_K, allowed_doc_ids=allowed_ids)
    except HTTPException:
        raise
    except urllib.error.URLError as exc:
        logger.error("Ollama not reachable at %s: %s", MODEL_BASE_URL, exc)
        raise HTTPException(status_code=503, detail=f"Ollama not reachable at {MODEL_BASE_URL}: {exc}")
    except Exception as exc:
        logger.exception("Error during retrieval")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    should_add_profile = False
    q_low = question.lower()
    if req.doc_ids and len(req.doc_ids) == 1:
        should_add_profile = True
    elif any(trigger in q_low for trigger in OVERVIEW_TRIGGERS):
        should_add_profile = True

    if should_add_profile and hits:
        if req.doc_ids and len(req.doc_ids) == 1:
            target_doc_id = req.doc_ids[0]
        else:
            doc_counts: Dict[str, int] = {}
            for hit in hits:
                doc_counts[hit["doc_id"]] = doc_counts.get(hit["doc_id"], 0) + 1
            target_doc_id = max(doc_counts, key=doc_counts.get)
        doc = doc_lookup.get(target_doc_id)
        if doc and doc.profile:
            if not any(h.get("kind") == "profile" and h["doc_id"] == target_doc_id for h in hits):
                hits = [
                    {
                        "text": doc.profile,
                        "page": 0,
                        "doc_id": target_doc_id,
                        "source": doc.filename,
                        "kind": "profile",
                    }
                ] + hits

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Retrieved candidates: %s",
            [
                (
                    h.get("source"),
                    h.get("page"),
                    h.get("kind", "chunk"),
                    (h["text"][:80] + "...") if len(h["text"]) > 80 else h["text"],
                )
                for h in hits
            ],
        )

    context_texts = [h["text"] for h in hits]
    prompt = render_prompt(context_texts, question)
    max_tokens = MAX_NEW_TOKENS
    if should_add_profile:
        max_tokens = max(max_tokens, 220)

    raw = ollama_generate(prompt, max_tokens=max_tokens, temperature=TEMPERATURE)
    answer = postprocess_answer(raw)
    logger.info("Chat response generated (%d source chunks)", len(hits))

    session: Optional[ChatSession] = None
    session_id = req.session_id
    if session_id:
        session = db.execute(
            select(ChatSession).where(
                ChatSession.session_id == session_id,
                ChatSession.user_id == current_user.id,
            )
        ).scalar_one_or_none()
    if session is None:
        session_id = session_id or uuid.uuid4().hex[:12]
        session = ChatSession(
            session_id=session_id,
            user_id=current_user.id,
            title=(question[:80] or "New chat"),
        )
        db.add(session)
        db.flush()

    user_message = ChatMessage(
        session_id=session.id,
        role="user",
        content=question,
        sources=None,
    )
    assistant_message = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=answer,
        sources=json.dumps([
            {"doc_id": h["doc_id"], "page": h["page"], "source": h["source"]}
            for h in hits
        ]),
    )
    db.add_all([user_message, assistant_message])
    db.commit()

    return {
        "response": answer,
        "sources": [
            {"doc_id": h["doc_id"], "page": h["page"], "source": h["source"]}
            for h in hits
        ],
        "session_id": session.session_id,
    }

@app.get("/docs")
def list_docs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    documents = get_accessible_documents(db, current_user)
    return {
        "docs": [serialize_document(doc) for doc in documents],
        "total_docs": len(documents),
    }
