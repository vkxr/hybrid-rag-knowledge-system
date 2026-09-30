import { createHash } from "node:crypto";
import { type Chunk, chunkPages, extractPages } from "./chunking";
import { type Queryable, toVectorLiteral } from "./db";
import type { Embedder } from "./embeddings";

const INSERT_BATCH = 100;

export interface IngestResult {
  documentId: string;
  name: string;
  chunks: number;
  pages: number;
  duplicate: boolean;
}

export async function ingestPdf(db: Queryable, embedder: Embedder, name: string, pdf: Uint8Array) {
  const pages = await extractPages(pdf);
  return ingestPages(db, embedder, name, pages, sha256(pdf));
}

/** Chunk, embed and store a document. Re-uploading the same file returns the existing document. */
export async function ingestPages(
  db: Queryable,
  embedder: Embedder,
  name: string,
  pages: string[],
  hash = sha256(pages.join("\f")),
): Promise<IngestResult> {
  const existing = await db.query<{ id: string; page_count: number; chunks: number }>(
    `SELECT d.id, d.page_count, count(c.id)::int AS chunks
       FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
      WHERE d.sha256 = $1 GROUP BY d.id`,
    [hash],
  );
  if (existing.rows[0]) {
    const row = existing.rows[0];
    return { documentId: row.id, name, chunks: row.chunks, pages: row.page_count, duplicate: true };
  }

  const chunks = await chunkPages(pages);
  if (chunks.length === 0) throw new Error(`No extractable text in ${name} (scanned PDF?)`);

  // Embed before writing anything, so an API failure leaves no half-ingested document.
  const embeddings = await embedder.embedDocuments(chunks.map((c) => c.content));

  const doc = await db.query<{ id: string }>(
    "INSERT INTO documents (name, sha256, page_count) VALUES ($1, $2, $3) RETURNING id",
    [name, hash, pages.length],
  );
  const documentId = doc.rows[0].id;
  try {
    for (let start = 0; start < chunks.length; start += INSERT_BATCH) {
      await insertChunks(db, documentId, chunks.slice(start, start + INSERT_BATCH),
        embeddings.slice(start, start + INSERT_BATCH));
    }
  } catch (err) {
    await db.query("DELETE FROM documents WHERE id = $1", [documentId]);
    throw err;
  }
  return { documentId, name, chunks: chunks.length, pages: pages.length, duplicate: false };
}

async function insertChunks(db: Queryable, documentId: string, chunks: Chunk[], embeddings: number[][]) {
  const values: string[] = [];
  const params: unknown[] = [documentId];
  chunks.forEach((c, i) => {
    const p = params.length;
    values.push(`($1, $${p + 1}, $${p + 2}, $${p + 3}, $${p + 4})`);
    params.push(c.chunkIndex, c.page, c.content, toVectorLiteral(embeddings[i]));
  });
  await db.query(
    `INSERT INTO chunks (document_id, chunk_index, page, content, embedding) VALUES ${values.join(", ")}`,
    params,
  );
}

function sha256(data: Uint8Array | string): string {
  return createHash("sha256").update(data).digest("hex");
}
