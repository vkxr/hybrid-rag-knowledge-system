import { PGlite } from "@electric-sql/pglite";
import { vector } from "@electric-sql/pglite/vector";
import { EMBEDDING_DIM } from "../lib/config";
import { migrate, type Queryable } from "../lib/db";
import type { Embedder } from "../lib/embeddings";

/** Real Postgres (WASM) with pgvector, in memory. Same SQL as production. */
export async function testDb(): Promise<Queryable & { close(): Promise<void> }> {
  const db = new PGlite({ extensions: { vector } });
  await migrate(db);
  return db as unknown as Queryable & { close(): Promise<void> };
}

/**
 * Deterministic bag-of-words embedder: texts sharing words get similar vectors.
 * Good enough to test retrieval plumbing without calling OpenAI.
 */
export class FakeEmbedder implements Embedder {
  calls = 0;

  private embed(text: string): number[] {
    const v = new Array(EMBEDDING_DIM).fill(0);
    for (const word of text.toLowerCase().match(/[a-z]{3,}/g) ?? []) {
      let h = 2166136261;
      for (const ch of word) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
      v[Math.abs(h) % EMBEDDING_DIM] += 1;
    }
    const norm = Math.hypot(...v) || 1;
    return v.map((x) => x / norm);
  }

  async embedDocuments(texts: string[]) {
    this.calls++;
    return texts.map((t) => this.embed(t));
  }

  async embedQuery(text: string) {
    this.calls++;
    return this.embed(text);
  }
}
