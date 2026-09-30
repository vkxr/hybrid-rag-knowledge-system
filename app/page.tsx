"use client";

import { type FormEvent, Fragment, useCallback, useEffect, useRef, useState } from "react";
import type { ChatEvent, Source } from "@/lib/answer";

type Doc = { id: string; name: string; pageCount: number; chunkCount: number };
type Turn = { question: string; answer: string; sources: Source[]; citations: number[]; error?: string };
type Mode = "hybrid" | "vector" | "keyword";

/** Render "[2]" markers as links to the matching source card. */
function AnswerText({ text, turn }: { text: string; turn: number }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <div className="answer">
      {parts.map((p, i) => {
        const m = p.match(/^\[(\d+)\]$/);
        return m ? (
          <a key={i} className="cite" href={`#t${turn}-s${m[1]}`}>{m[1]}</a>
        ) : (
          <Fragment key={i}>{p}</Fragment>
        );
      })}
    </div>
  );
}

export default function Home() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("hybrid");
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const loadDocs = useCallback(async () => {
    const res = await fetch("/api/documents");
    if (res.ok) setDocs((await res.json()).documents);
  }, []);

  useEffect(() => { loadDocs(); }, [loadDocs]);

  async function upload(e: FormEvent) {
    e.preventDefault();
    const files = fileRef.current?.files;
    if (!files?.length) return;
    const form = new FormData();
    for (const f of files) form.append("files", f);
    setUploading(true);
    setUploadMsg("Extracting, chunking and embedding...");
    try {
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const { results } = await res.json();
      setUploadMsg(results.map((r: any) =>
        r.error ? `${r.name}: ${r.error}` : `${r.name}: ${r.chunks} chunks${r.duplicate ? " (already indexed)" : ""}`,
      ).join("\n"));
      if (fileRef.current) fileRef.current.value = "";
      await loadDocs();
    } catch {
      setUploadMsg("Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function remove(id: string) {
    await fetch(`/api/documents/${id}`, { method: "DELETE" });
    setSelected((s) => { const n = new Set(s); n.delete(id); return n; });
    await loadDocs();
  }

  function toggle(id: string) {
    setSelected((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  }

  const update = (i: number, fn: (t: Turn) => Turn) =>
    setTurns((ts) => ts.map((t, j) => (j === i ? fn(t) : t)));

  async function ask(e: FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    const index = turns.length;
    setTurns((ts) => [...ts, { question: q, answer: "", sources: [], citations: [] }]);
    setQuestion("");
    setBusy(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, mode, documentIds: selected.size ? [...selected] : undefined }),
      });
      if (!res.ok || !res.body) throw new Error(`Request failed (${res.status})`);
      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += value;
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines.filter(Boolean)) {
          const ev = JSON.parse(line) as ChatEvent;
          if (ev.type === "sources") update(index, (t) => ({ ...t, sources: ev.sources }));
          if (ev.type === "token") update(index, (t) => ({ ...t, answer: t.answer + ev.text }));
          if (ev.type === "done") update(index, (t) => ({ ...t, citations: ev.citations }));
          if (ev.type === "error") update(index, (t) => ({ ...t, error: ev.message }));
        }
      }
    } catch (err) {
      update(index, (t) => ({ ...t, error: err instanceof Error ? err.message : "Something went wrong" }));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="layout">
      <aside className="panel">
        <h1>Knowledge base</h1>
        <p className="muted">Upload PDFs, then ask questions across them.</p>
        <form onSubmit={upload}>
          <input ref={fileRef} type="file" accept="application/pdf" multiple />
          <div style={{ marginTop: 8 }}>
            <button disabled={uploading}>{uploading ? "Indexing..." : "Upload"}</button>
          </div>
        </form>
        {uploadMsg && <p className="muted" style={{ whiteSpace: "pre-wrap" }}>{uploadMsg}</p>}

        <h2>Documents {selected.size > 0 && `(${selected.size} selected)`}</h2>
        {docs.length === 0 && <p className="muted">No documents yet.</p>}
        {docs.map((d) => (
          <div className="doc" key={d.id}>
            <input id={d.id} type="checkbox" checked={selected.has(d.id)} onChange={() => toggle(d.id)} />
            <label htmlFor={d.id}>
              {d.name}
              <div className="muted">{d.pageCount} pages, {d.chunkCount} chunks</div>
            </label>
            <button className="ghost" onClick={() => remove(d.id)} aria-label={`Delete ${d.name}`}>✕</button>
          </div>
        ))}
        <p className="muted">No selection searches every document.</p>
      </aside>

      <section className="panel chat">
        <div className="messages">
          {turns.length === 0 && <p className="muted">Ask something about your documents.</p>}
          {turns.map((t, i) => (
            <div key={i}>
              <div className="question">{t.question}</div>
              {t.answer ? <AnswerText text={t.answer} turn={i} /> : !t.error && <p className="muted">Searching...</p>}
              {t.error && <p className="error">{t.error}</p>}
              {t.sources.length > 0 && (
                <div className="sources">
                  {t.sources.map((s, j) => (
                    <div id={`t${i}-s${j + 1}`} key={s.id}
                      className={`source${t.citations.includes(j + 1) ? " cited" : ""}`}>
                      <strong>[{j + 1}] {s.documentName}, p. {s.page}</strong>
                      <p>{s.snippet}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
        <form className="composer" onSubmit={ask}>
          <textarea value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask a question"
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) ask(e); }} />
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)} aria-label="Retrieval mode">
            <option value="hybrid">Hybrid</option>
            <option value="vector">Vector</option>
            <option value="keyword">Keyword</option>
          </select>
          <button disabled={busy || !question.trim()}>Ask</button>
        </form>
      </section>
    </main>
  );
}
