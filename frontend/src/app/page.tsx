"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type SourceInfo = { doc_id: string; page: number; source: string };

type ChatMessage =
  | { role: "user"; content: string }
  | {
      role: "model";

      content: string;

      sources?: SourceInfo[];
    };

type DocInfo = {
  doc_id: string;

  filename: string;

  pages: number;

  chunk_count: number;

  uploaded_at: string;

  section_count?: number;

  profile?: string;
};

function groupSources(sources?: SourceInfo[]) {
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

function isSourceInfoArray(value: unknown): value is SourceInfo[] {
  return (
    Array.isArray(value) &&
    value.every(
      (item) =>
        item &&
        typeof item === "object" &&
        typeof (item as Record<string, unknown>).doc_id === "string" &&
        typeof (item as Record<string, unknown>).source === "string" &&
        typeof (item as Record<string, unknown>).page === "number",
    )
  );
}

export default function Home() {
  const [token, setToken] = useState<string | null>(null);

  const [authMode, setAuthMode] = useState<"login" | "signup">("login");

  const [email, setEmail] = useState<string>("");

  const [password, setPassword] = useState<string>("");

  const [isAuthLoading, setIsAuthLoading] = useState<boolean>(false);

  const [authError, setAuthError] = useState<string>("");

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

  const [sessionId, setSessionId] = useState<string | null>(null);

  // Status

  const [error, setError] = useState<string>("");

  const [banner, setBanner] = useState<string>("");

  const [toast, setToast] = useState<string>("");

  const handleUnauthorized = (
    message: string = "Session expired. Please log in again.",
  ) => {
    setAuthError(message);

    setToken(null);

    setDocs([]);

    setSelectedDocIds([]);

    setChatHistory([]);

    setSessionId(null);
  };

  // Backend URL (same host convenience)

  const backendHost =
    typeof window !== "undefined" ? window.location.hostname : "localhost";

  const backendUrl = useMemo(() => `http://${backendHost}:8000`, [backendHost]);

  // Refs

  const chatBottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, isChatting]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const stored = localStorage.getItem("rag_token");

    if (stored) {
      setToken(stored);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;

    if (token) {
      localStorage.setItem("rag_token", token);
    } else {
      localStorage.removeItem("rag_token");
    }
  }, [token]);

  // Fetch docs on load

  const fetchDocs = useCallback(async () => {
    if (!token) return;

    try {
      const res = await fetch(`${backendUrl}/docs`, {
        cache: "no-store",

        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (res.status === 401) {
        handleUnauthorized();

        return;
      }

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const raw = await res.json().catch(() => null);

      const data =
        raw && typeof raw === "object" ? (raw as { docs?: DocInfo[] }) : null;

      setDocs(Array.isArray(data?.docs) ? data.docs : []);
    } catch (err) {
      console.error("GET /docs failed:", err);

      setDocs([]);
    }
  }, [backendUrl, token]);

  useEffect(() => {
    if (!token) {
      setDocs([]);

      return;
    }

    void fetchDocs();
  }, [token, fetchDocs]);


  const resetForNewUpload = () => {
    setChatHistory([]);

    setUploadStatus("");

    setError("");

    setBanner("");

    setSessionId(null);
  };

  const handleFileInput = (f: File | null) => {
    setFile(f);

    if (f) resetForNewUpload();
  };

  // === Auth ===

  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    setIsAuthLoading(true);

    setAuthError("");

    setError("");

    try {
      const endpoint = authMode === "login" ? "/auth/login" : "/auth/signup";

      const res = await fetch(`${backendUrl}${endpoint}`, {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({ email, password }),
      });

      const raw = await res.json().catch(() => null);

      const data =
        raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};

      if (!res.ok) {
        const detail =
          typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`;

        setAuthError(detail);

        return;
      }

      const accessToken =
        typeof data.access_token === "string" ? data.access_token : null;

      if (!accessToken) {
        setAuthError("Malformed authentication response.");

        return;
      }

      setAuthError("");

      setToken(accessToken);

      setToast(authMode === "login" ? "Logged in." : "Account created.");

      setEmail("");

      setPassword("");

      setBanner("");

      await fetchDocs();
    } catch (err) {
      console.error(err);

      setAuthError("Authentication failed.");
    } finally {
      setIsAuthLoading(false);
    }
  };

  const handleLogout = () => {
    setToken(null);

    setDocs([]);

    setSelectedDocIds([]);

    setChatHistory([]);

    setSessionId(null);

    setAuthError("");

    setEmail("");

    setPassword("");

    setToast("");

    setError("");

    setBanner("");

    setFile(null);

    setUploadStatus("");
  };

  // === Upload (stage only, no indexing) ===

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!token) {
      setAuthError("Please log in to upload documents.");

      return;
    }

    if (!file) {
      setError("Please select a PDF first.");

      return;
    }

    setError("");

    setBanner("");

    setIsUploading(true);

    setUploadStatus("Uploading...");

    const formData = new FormData();

    formData.append("file", file);

    try {
      const res = await fetch(`${backendUrl}/upload`, {
        method: "POST",

        body: formData,

        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await res.json();

      if (res.ok) {
        setUploadStatus(
          data.status === "exists"
            ? `Already added: ${data.filename}`
            : `Added: ${data.filename}`,
        );

        await fetchDocs();
      } else {
        setUploadStatus("");

        if (res.status === 401) {
          handleUnauthorized();

          return;
        }

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
    if (!token) {
      setAuthError("Please log in to modify documents.");

      return;
    }

    try {
      const res = await fetch(`${backendUrl}/docs/${doc_id}`, {
        method: "DELETE",

        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await res.json();

      if (res.ok) {
        setToast(
          `Removed ${data.removed} file(s). Click "Build Index" to apply.`,
        );

        if (selectedDocIds.includes(doc_id)) {
          setSelectedDocIds((prev) => prev.filter((d) => d !== doc_id));
        }

        setSessionId(null);

        await fetchDocs();
      } else {
        if (res.status === 401) {
          handleUnauthorized();

          return;
        }

        setError(data?.detail || "Delete failed");
      }
    } catch (err) {
      console.error("DELETE /docs failed:", err);

      setError("Delete failed.");
    }
  };

  // === Build / Rebuild the index ===

  const buildIndex = async () => {
    if (!token) {
      setAuthError("Please log in to build indexes.");

      return;
    }

    setIsIndexing(true);

    setToast("");

    setError("");

    try {
      const body =
        selectedDocIds.length > 0
          ? { doc_ids: selectedDocIds }
          : { doc_ids: null };

      const res = await fetch(`${backendUrl}/index`, {
        method: "POST",

        headers: {
          "Content-Type": "application/json",

          Authorization: `Bearer ${token}`,
        },

        body: JSON.stringify(body),
      });

      const data = await res.json();

      if (res.ok) {
        setToast(
          `Indexed ${data.docs_indexed} document(s) - ${data.total_chunks} chunks`,
        );

        setSessionId(null);
      } else {
        if (res.status === 401) {
          handleUnauthorized();

          return;
        }

        setError(data?.detail || "Indexing failed");
      }
    } catch (err) {
      console.error("POST /index failed:", err);

      setError("Indexing failed.");
    } finally {
      setIsIndexing(false);
    }
  };

  const toggleDoc = (id: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id],
    );

    setSessionId(null);

    setChatHistory([]);
  };

  const clearDocSelection = () => {
    setSelectedDocIds([]);

    setSessionId(null);

    setChatHistory([]);
  };

  // === Chat ===

  const sendMessage = async (text: string) => {
    if (!token) {
      setAuthError("Please log in to chat.");

      return;
    }

    setIsChatting(true);

    setError("");

    setBanner("");

    const newHistory: ChatMessage[] = [
      ...chatHistory,
      { role: "user", content: text },
    ];

    setChatHistory(newHistory);

    try {
      const body: Record<string, unknown> = { message: text };

      if (selectedDocIds.length > 0) body.doc_ids = selectedDocIds;

      if (sessionId) body.session_id = sessionId;

      const res = await fetch(`${backendUrl}/chat`, {
        method: "POST",

        headers: {
          "Content-Type": "application/json",

          Authorization: `Bearer ${token}`,
        },

        body: JSON.stringify(body),
      });

      const raw = await res.json().catch(() => null);

      const data =
        raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};

      if (res.ok) {
        const responseText =
          typeof data.response === "string" ? data.response : "No response.";

        const sources = isSourceInfoArray(data.sources)
          ? data.sources
          : undefined;

        const modelMessage: ChatMessage = {
          role: "model",
          content: responseText,
        };

        if (sources && sources.length > 0) {
          modelMessage.sources = sources;
        }

        setChatHistory([...newHistory, modelMessage]);

        if (typeof data.session_id === "string") {
          setSessionId(data.session_id);
        }
      } else {
        if (res.status === 401) {
          handleUnauthorized();

          return;
        }

        if (res.status === 503) {
          setBanner("Index not ready. Add files and click Build Index first.");
        }

        const detail =
          typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`;

        setChatHistory([
          ...newHistory,
          { role: "model", content: `Error: ${detail}` },
        ]);

        setError(detail);
      }
    } catch (err) {
      const msg = "Failed to reach server.";

      setChatHistory([
        ...newHistory,
        { role: "model", content: `Error: ${msg}` },
      ]);

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

  if (!token) {
    return (
      <main className="min-h-screen w-full bg-gradient-to-b from-black via-zinc-950 to-black text-zinc-100 flex items-center justify-center p-6">
        <div className="w-full max-w-sm rounded-2xl border border-white/10 bg-white/5 p-6 shadow-xl shadow-black/40 space-y-4">
          <h1 className="text-lg font-semibold text-center">
            Local RAG Engine
          </h1>

          <p className="text-sm text-zinc-400 text-center">
            {authMode === "login"
              ? "Sign in to continue."
              : "Create an account to get started."}
          </p>

          {authError && (
            <div className="text-xs text-red-300/90 text-center bg-red-500/10 border border-red-400/30 rounded-lg px-3 py-2">
              {authError}
            </div>
          )}

          <form onSubmit={handleAuthSubmit} className="space-y-3">
            <div className="space-y-1">
              <label className="text-xs uppercase text-zinc-400">Email</label>

              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                required
              />
            </div>

            <div className="space-y-1">
              <label className="text-xs uppercase text-zinc-400">
                Password
              </label>

              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                required
              />
            </div>

            <button
              type="submit"
              disabled={isAuthLoading}
              className="w-full rounded-lg bg-gradient-to-r from-indigo-600 to-violet-600 py-2 text-sm font-semibold text-white shadow-lg shadow-indigo-900/30 disabled:from-zinc-700 disabled:to-zinc-700 disabled:shadow-none"
            >
              {isAuthLoading
                ? "Submitting..."
                : authMode === "login"
                  ? "Log In"
                  : "Sign Up"}
            </button>
          </form>

          <button
            onClick={() =>
              setAuthMode((prev) => (prev === "login" ? "signup" : "login"))
            }
            className="w-full text-xs text-zinc-400 hover:text-zinc-200"
            type="button"
          >
            {authMode === "login"
              ? "Need an account? Sign up"
              : "Have an account? Log in"}
          </button>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen w-full bg-gradient-to-b from-black via-zinc-950 to-black text-zinc-100">
      {/* Header */}

      <header className="sticky top-0 z-10 backdrop-blur supports-[backdrop-filter]:bg-black/40 bg-black/30 border-b border-white/10">
        <div className="mx-auto max-w-6xl px-4 py-4 flex items-center justify-between">
          <div className="text-xs text-zinc-500">Local | Private | Fast</div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleLogout}
              className="text-[11px] rounded-md border border-white/10 bg-white/10 px-2 py-1 text-zinc-300 hover:bg-white/20"
            >
              Log Out
            </button>

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
                    ? "Building..."
                    : selectedDocIds.length > 0
                      ? "Build Index (Selected)"
                      : "Build Index (All)"}
                </button>

                <button
                  onClick={() => setSelectedDocIds(docs.map((d) => d.doc_id))}
                  disabled={docs.length === 0}
                  className="text-[11px] px-2 py-1 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 disabled:opacity-50"
                  title="Select all"
                >
                  Select All
                </button>

                <button
                  onClick={clearDocSelection}
                  disabled={selectedDocIds.length === 0}
                  className="text-[11px] px-2 py-1 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 disabled:opacity-50"
                  title="Clear selection"
                >
                  Clear
                </button>

                <span className="text-[11px] text-zinc-400">
                  {docs.length} file{docs.length !== 1 ? "s" : ""}
                </span>
              </div>
            </div>

            {/* Cards grid */}

            {docs.length === 0 ? (
              <div className="text-[12px] text-zinc-400">
                No files yet. Upload below.
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {docs.map((d) => {
                  const active = selectedDocIds.includes(d.doc_id);

                  const uploaded = (() => {
                    try {
                      return new Date(d.uploaded_at).toLocaleString();
                    } catch {
                      return d.uploaded_at;
                    }
                  })();

                  const snippet = (d.profile || "").slice(0, 120);

                  return (
                    <div
                      key={d.doc_id}
                      className={`rounded-xl border p-3 transition shadow-sm ${
                        active
                          ? "border-indigo-400/40 bg-indigo-500/10"
                          : "border-white/10 bg-black/30"
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        <input
                          type="checkbox"
                          checked={active}
                          onChange={() => toggleDoc(d.doc_id)}
                          className="mt-1 h-4 w-4 accent-indigo-500"
                          aria-label={`Select ${d.filename}`}
                        />

                        <div className="min-w-0 flex-1">
                          <div
                            className="truncate text-sm font-medium text-zinc-100"
                            title={d.filename}
                          >
                            {d.filename}
                          </div>

                          <div className="mt-0.5 text-[11px] text-zinc-400">
                            {d.pages ?? 0}p · {d.section_count ?? 0}s ·{" "}
                            {d.chunk_count ?? 0}c
                          </div>

                          <div className="mt-0.5 text-[11px] text-zinc-500">
                            Uploaded {uploaded}
                          </div>

                          {snippet && (
                            <div className="mt-2 text-[12px] text-zinc-300 line-clamp-3">
                              {snippet}

                              {d.profile && d.profile.length > 120 ? "..." : ""}
                            </div>
                          )}
                        </div>

                        <button
                          onClick={() => deleteDoc(d.doc_id)}
                          className="ml-2 shrink-0 text-[11px] rounded-md border border-white/10 px-2 py-1 hover:bg-white/10"
                          title="Remove from library"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
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

                      <div className="text-xs text-zinc-400">
                        Click to replace
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="font-medium">Click to select a PDF</div>

                      <div className="text-xs text-zinc-400">
                        Only .pdf is accepted
                      </div>
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
              {isUploading ? "Adding..." : "Add to Library"}
            </button>

            {uploadStatus && (
              <p className="mt-3 text-xs text-emerald-300/90">{uploadStatus}</p>
            )}

            {banner && (
              <p className="mt-3 text-xs text-amber-300/90">{banner}</p>
            )}

            {error && <p className="mt-3 text-xs text-red-300/90">{error}</p>}

            <p className="mt-3 text-[11px] text-zinc-400">
              After adding files, click{" "}
              <span className="text-zinc-300">Build Index</span> to refresh
              search.
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
                  Filtering {selectedDocIds.length} doc
                  {selectedDocIds.length !== 1 ? "s" : ""}
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

                    {msg.role === "model" &&
                      Array.isArray(msg.sources) &&
                      msg.sources.length > 0 && (
                        <div className="mt-3 border-t border-white/10 pt-2">
                          <div className="text-[11px] text-zinc-400 mb-1">
                            Sources
                          </div>

                          <div className="flex flex-col gap-1.5">
                            {groupSources(msg.sources).map((g, idx) => (
                              <div
                                key={idx}
                                className="text-[12px] text-zinc-300"
                              >
                                <span className="font-medium">{g.file}</span>

                                <span className="text-zinc-500"> · p.</span>

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
                    <span className="inline-flex animate-pulse">
                      Thinking...
                    </span>
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
                  placeholder="Ask a question... (Shift+Enter for newline)"
                  rows={2}
                  className="flex-1 resize-none rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-sm outline-none ring-0 placeholder:text-zinc-500"
                />

                <button
                  type="submit"
                  disabled={isChatting || message.trim().length === 0}
                  className="h-10 shrink-0 rounded-xl bg-gradient-to-br from-emerald-600 to-green-600 px-4 text-sm font-semibold text-white shadow-lg shadow-emerald-900/30 disabled:from-zinc-700 disabled:to-zinc-700 disabled:shadow-none"
                >
                  {isChatting ? "Sending..." : "Send"}
                </button>
              </div>

              <div className="mt-2 flex items-center justify-between text-[11px] text-zinc-500">
                <span>Enter to send · Shift+Enter for newline</span>
              </div>
            </form>
          </div>
        </section>
      </div>
    </main>
  );
}
