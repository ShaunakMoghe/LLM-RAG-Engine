"use client";
import { useEffect, useMemo, useRef, useState } from "react";

type ChatMessage =
  | { role: "user"; content: string }
  | {
      role: "model";
      content: string;
      sources?: Array<{ doc_id: string; page: number; source: string }>;
    };

type DocInfo = {
  doc_id: string;
  filename: string;
  pages: number;
  chunk_count: number;
  uploaded_at: string;
};

function groupSources(
  sources?: Array<{ doc_id: string; page: number; source: string }>
) {
  if (!sources || sources.length === 0) return [];
  const byFile: Record<string, number[]> = {};
  for (const s of sources) {
    const key = s.source || "unknown";
    (byFile[key] ||= []).push(s.page);
  }
  return Object.entries(byFile).map(([file, pages]) => ({
    file,
    pages: Array.from(new Set(pages)).sort((a, b) => a - b),
  }));
}

export default function Home() {
  // Upload state
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string>("");
  const [isUploading, setIsUploading] = useState<boolean>(false);

  // Docs / Library
  const [docs, setDocs] = useState<DocInfo[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]); // empty = all
  const [isIndexing, setIsIndexing] = useState<boolean>(false);

  // Chat state
  const [message, setMessage] = useState<string>("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isChatting, setIsChatting] = useState<boolean>(false);

  // Status
  const [error, setError] = useState<string>("");
  const [banner, setBanner] = useState<string>("");
  const [toast, setToast] = useState<string>("");

  // Backend URL (same host convenience)
  const backendHost =
    typeof window !== "undefined" ? window.location.hostname : "localhost";
  const backendUrl = useMemo(() => `http://${backendHost}:8000`, [backendHost]);

  // Refs
  const chatBottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, isChatting]);

  // Fetch docs on load
  useEffect(() => {
    void fetchDocs();
  }, []);

  const fetchDocs = async () => {
    try {
      const res = await fetch(`${backendUrl}/docs`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setDocs(data?.docs || []);
    } catch (e) {
      console.error("GET /docs failed:", e);
      setDocs([]);        // show 0, but now you’ll see the error in the console
    }
  };


  const resetForNewUpload = () => {
    setChatHistory([]);
    setUploadStatus("");
    setError("");
    setBanner("");
  };

  const handleFileInput = (f: File | null) => {
    setFile(f);
    if (f) resetForNewUpload();
  };

  // === Upload (stage only, no indexing) ===
  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a PDF first.");
      return;
    }
    setError("");
    setBanner("");
    setIsUploading(true);
    setUploadStatus("Uploading…");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${backendUrl}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        console.log("UPLOAD OK:", data); // should show { status: "staged"/"exists", doc_id, ... }
        setUploadStatus(
          data.status === "exists" ? `Already added: ${data.filename}` : `Added: ${data.filename}`
        );
        await fetchDocs();  // should now repopulate Library
      } else {
        setUploadStatus("");
        setError(data?.detail || "Upload failed");
      }
    } catch (err) {
      setUploadStatus("");
      setError("Upload failed. Is the backend running on :8000?");
      console.error(err);
    } finally {
      setIsUploading(false);
    }
  };

  // === Delete a doc from library ===
  const deleteDoc = async (doc_id: string) => {
    try {
      const res = await fetch(`${backendUrl}/docs/${doc_id}`, { method: "DELETE" });
      const data = await res.json();
      if (res.ok) {
        setToast(`Removed ${data.removed} file(s). Click “Build Index” to apply.`);
        if (selectedDocIds.includes(doc_id)) {
          setSelectedDocIds((prev) => prev.filter((d) => d !== doc_id));
        }
        await fetchDocs();
      } else {
        setError(data?.detail || "Delete failed");
      }
    } catch (e) {
      setError("Delete failed.");
    }
  };

  // === Build / Rebuild the index ===
  const buildIndex = async () => {
    setIsIndexing(true);
    setToast("");
    setError("");
    try {
      const body =
        selectedDocIds.length > 0 ? { doc_ids: selectedDocIds } : { doc_ids: null };
      const res = await fetch(`${backendUrl}/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (res.ok) {
        setToast(
          `Indexed ${data.docs_indexed} document(s) • ${data.total_chunks} chunks`
        );
      } else {
        setError(data?.detail || "Indexing failed");
      }
    } catch (e) {
      setError("Indexing failed.");
    } finally {
      setIsIndexing(false);
    }
  };

  const toggleDoc = (id: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id]
    );
  };

  const clearDocSelection = () => setSelectedDocIds([]);

  // === Chat ===
  const sendMessage = async (text: string) => {
    setIsChatting(true);
    setError("");
    setBanner("");

    const newHistory: ChatMessage[] = [...chatHistory, { role: "user", content: text }];
    setChatHistory(newHistory);

    try {
      const body: any = { message: text };
      if (selectedDocIds.length > 0) body.doc_ids = selectedDocIds;

      const res = await fetch(`${backendUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await res.json().catch(() => ({} as any));

      if (res.ok) {
        const sources = (data as any)?.sources as ChatMessage extends {
          role: "model";
          sources: infer S;
        }
          ? S
          : any;
        setChatHistory([...newHistory, { role: "model", content: data.response, sources }]);
      } else {
        const detail = (data as any)?.detail || `HTTP ${res.status}`;
        if (res.status === 503) {
          setBanner(
            "Index not ready. Add files and click “Build Index” first."
          );
        }
        setChatHistory([...newHistory, { role: "model", content: `Error: ${detail}` }]);
        setError(detail);
      }
    } catch (err) {
      const msg = "Failed to reach server.";
      setChatHistory([...newHistory, { role: "model", content: `Error: ${msg}` }]);
      setError(msg);
      console.error(err);
    } finally {
      setIsChatting(false);
    }
  };

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = message.trim();
    if (!trimmed || isChatting) return;
    await sendMessage(trimmed);
    setMessage("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const trimmed = message.trim();
      if (trimmed && !isChatting) {
        void sendMessage(trimmed);
        setMessage("");
      }
    }
  };

  const allDocsSelected = selectedDocIds.length === 0;

  return (
    <main className="min-h-screen w-full bg-gradient-to-b from-black via-zinc-950 to-black text-zinc-100">
      {/* Header */}
      <header className="sticky top-0 z-10 backdrop-blur supports-[backdrop-filter]:bg-black/40 bg-black/30 border-b border-white/10">
        <div className="mx-auto max-w-6xl px-4 py-4 flex items-center justify-between">
          <div className="text-xs text-zinc-500">Local • Private • Fast</div>
          <div className="flex items-center gap-2">
            {toast && (
              <span className="text-[11px] rounded-md border border-emerald-400/30 bg-emerald-500/10 px-2 py-1 text-emerald-200">
                {toast}
              </span>
            )}
            {error && (
              <span className="text-[11px] rounded-md border border-red-400/30 bg-red-500/10 px-2 py-1 text-red-200">
                {error}
              </span>
            )}
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-6xl px-4 py-6 grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left: Library + Upload */}
        <section className="lg:col-span-2 space-y-6">
          {/* Library */}
          <div className="rounded-2xl border border-white/10 bg-white/5 p-5 shadow-xl shadow-black/30">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-base font-semibold">Library</h2>
              <div className="flex items-center gap-2">
                <button
                  onClick={buildIndex}
                  disabled={isIndexing || docs.length === 0}
                  className="text-[11px] px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white disabled:bg-zinc-700"
                  title={
                    selectedDocIds.length > 0
                      ? "Build index for selected docs"
                      : "Build index for all docs"
                  }
                >
                  {isIndexing
                    ? "Building…"
                    : selectedDocIds.length > 0
                    ? "Build Index (Selected)"
                    : "Build Index (All)"}
                </button>
                {!allDocsSelected && (
                  <button
                    onClick={clearDocSelection}
                    className="text-[11px] px-2 py-1 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10"
                  >
                    Clear
                  </button>
                )}
                <span className="text-[11px] text-zinc-400">
                  {docs.length} file{docs.length !== 1 ? "s" : ""}
                </span>
              </div>
            </div>

            {/* Doc chips */}
            <div className="flex flex-col gap-2">
              {docs.length === 0 && (
                <div className="text-[12px] text-zinc-400">No files yet. Upload below.</div>
              )}

              {docs.map((d) => {
                const active = selectedDocIds.includes(d.doc_id);
                return (
                  <div
                    key={d.doc_id}
                    className={`flex items-center justify-between rounded-xl px-3 py-2 ring-1 ${
                      active
                        ? "bg-indigo-500/15 ring-indigo-400/30"
                        : "bg-black/30 ring-white/10"
                    }`}
                  >
                    <button
                      onClick={() => toggleDoc(d.doc_id)}
                      className="flex-1 text-left truncate"
                      title={d.filename}
                    >
                      <div className="text-xs font-medium text-zinc-100 truncate">
                        {d.filename}
                      </div>
                      <div className="text-[11px] text-zinc-400">
                        {d.pages}p • {d.chunk_count}c
                      </div>
                    </button>
                    <button
                      onClick={() => deleteDoc(d.doc_id)}
                      className="ml-2 shrink-0 text-[11px] rounded-md border border-white/10 px-2 py-1 hover:bg-white/10"
                      title="Remove from library"
                    >
                      Delete
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Upload */}
          <div className="rounded-2xl border border-white/10 bg-white/5 p-5 shadow-xl shadow-black/30">
            <h2 className="text-base font-semibold mb-3">Upload</h2>

            <label htmlFor="pdf" className="block">
              <div className="group cursor-pointer rounded-xl border border-dashed border-white/15 bg-black/30 hover:bg-black/40 transition p-5 text-center">
                <input
                  id="pdf"
                  type="file"
                  accept=".pdf"
                  className="hidden"
                  onChange={(e) => handleFileInput(e.target.files?.[0] || null)}
                />
                <div className="text-sm text-zinc-300">
                  {file ? (
                    <>
                      <div className="truncate font-medium text-zinc-100">
                        {file.name}
                      </div>
                      <div className="text-xs text-zinc-400">Click to replace</div>
                    </>
                  ) : (
                    <>
                      <div className="font-medium">Click to select a PDF</div>
                      <div className="text-xs text-zinc-400">Only .pdf is accepted</div>
                    </>
                  )}
                </div>
              </div>
            </label>

            <button
              onClick={handleUpload}
              disabled={!file || isUploading}
              className="mt-4 inline-flex w-full items-center justify-center rounded-xl bg-gradient-to-br from-indigo-600 to-violet-600 px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-900/30 disabled:from-zinc-700 disabled:to-zinc-700 disabled:shadow-none"
            >
              {isUploading ? "Adding…" : "Add to Library"}
            </button>

            {uploadStatus && (
              <p className="mt-3 text-xs text-emerald-300/90">{uploadStatus}</p>
            )}
            {banner && (
              <p className="mt-3 text-xs text-amber-300/90">{banner}</p>
            )}
            {error && <p className="mt-3 text-xs text-red-300/90">{error}</p>}

            <p className="mt-3 text-[11px] text-zinc-400">
              After adding files, click <span className="text-zinc-300">Build Index</span> to
              refresh search.
            </p>
          </div>
        </section>

        {/* Right: Chat */}
        <section className="lg:col-span-3">
          <div className="rounded-2xl border border-white/10 bg-white/5 p-5 shadow-xl shadow-black/30 flex flex-col h-[70vh]">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-base font-semibold">Chat</h2>
              {selectedDocIds.length > 0 ? (
                <span className="text-[11px] text-zinc-400">
                  Filtering {selectedDocIds.length} doc{selectedDocIds.length !== 1 ? "s" : ""}
                </span>
              ) : (
                <span className="text-[11px] text-zinc-400">All documents</span>
              )}
            </div>

            {/* Inline banner for backend issues */}
            {banner && (
              <div className="mb-3 rounded-lg border border-amber-400/40 bg-amber-50/10 px-3 py-2 text-xs text-amber-200">
                {banner}
              </div>
            )}

            <div className="flex-1 overflow-y-auto rounded-xl bg-black/30 border border-white/10 p-4 space-y-3">
              {chatHistory.length === 0 && (
                <div className="text-center text-sm text-zinc-400 py-10">
                  Add files, build the index, then ask a question.
                </div>
              )}

              {chatHistory.map((msg, i) => (
                <div
                  key={i}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow ${
                      msg.role === "user"
                        ? "bg-gradient-to-br from-indigo-600 to-violet-600 text-white"
                        : "bg-white/10 text-zinc-100"
                    }`}
                  >
                    <div>{msg.content}</div>

                    {/* Grouped Sources */}
                    {msg.role === "model" && (msg as any).sources?.length > 0 && (
                      <div className="mt-3 border-t border-white/10 pt-2">
                        <div className="text-[11px] text-zinc-400 mb-1">Sources</div>
                        <div className="flex flex-col gap-1.5">
                          {groupSources((msg as any).sources).map((g, idx) => (
                            <div key={idx} className="text-[12px] text-zinc-300">
                              <span className="font-medium">{g.file}</span>
                              <span className="text-zinc-500"> — p.</span>
                              <span>{g.pages.join(", ")}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {isChatting && (
                <div className="flex justify-start">
                  <div className="max-w-[80%] rounded-2xl px-4 py-3 text-sm bg-white/10 text-zinc-100 shadow">
                    <span className="inline-flex animate-pulse">Thinking…</span>
                  </div>
                </div>
              )}
              <div ref={chatBottomRef} />
            </div>

            <form onSubmit={handleChatSubmit} className="mt-4">
              <div className="flex items-end gap-2">
                <textarea
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a question… (Shift+Enter for newline)"
                  rows={2}
                  className="flex-1 resize-none rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-sm outline-none ring-0 placeholder:text-zinc-500"
                />
                <button
                  type="submit"
                  disabled={isChatting || message.trim().length === 0}
                  className="h-10 shrink-0 rounded-xl bg-gradient-to-br from-emerald-600 to-green-600 px-4 text-sm font-semibold text-white shadow-lg shadow-emerald-900/30 disabled:from-zinc-700 disabled:to-zinc-700 disabled:shadow-none"
                >
                  {isChatting ? "Sending…" : "Send"}
                </button>
              </div>
              <div className="mt-2 flex items-center justify-between text-[11px] text-zinc-500">
                <span>Enter to send • Shift+Enter for newline</span>
              </div>
            </form>
          </div>
        </section>
      </div>
    </main>
  );
}
