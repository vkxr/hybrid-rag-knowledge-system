import { CANDIDATES_PER_RETRIEVER, RRF_K, TOP_K } from "./config";
import { type Queryable, toVectorLiteral } from "./db";
import type { Embedder } from "./embeddings";

export type RetrievalMode = "hybrid" | "vector" | "keyword";

export interface RetrievedChunk {
  id: number;
  documentId: string;
  documentName: string;
  chunkIndex: number;
  page: number;
  content: string;
  score: number; // RRF score in hybrid mode, similarity / rank score otherwise
  vectorRank?: number; // 1-based rank from each retriever, if it returned this chunk
  keywordRank?: number;
}

interface SearchOptions {
  limit?: number;
  documentIds?: string[]; // restrict to these documents; omit to search everything
}

const SELECT = `c.id::int AS id, c.document_id AS "documentId", d.name AS "documentName",
  c.chunk_index AS "chunkIndex", c.page, c.content`;

const DOC_FILTER = "($2::uuid[] IS NULL OR c.document_id = ANY($2::uuid[]))";

/** Cosine similarity over pgvector (HNSW index). */
export async function vectorSearch(db: Queryable, embedding: number[], opts: SearchOptions = {}) {
  const { rows } = await db.query<RetrievedChunk>(
    `SELECT ${SELECT}, 1 - (c.embedding <=> $1::vector) AS score
       FROM chunks c JOIN documents d ON d.id = c.document_id
      WHERE ${DOC_FILTER}
      ORDER BY c.embedding <=> $1::vector
      LIMIT $3`,
    [toVectorLiteral(embedding), opts.documentIds?.length ? opts.documentIds : null,
      opts.limit ?? CANDIDATES_PER_RETRIEVER],
  );
  return rows.map((r) => ({ ...r, score: Number(r.score) }));
}

/**
 * PostgreSQL full-text search (GIN index), ranked by cover density.
 *
 * websearch_to_tsquery ANDs every term, so a natural question ("how many days for a refund")
 * matches nothing unless one chunk contains every word. We parse with websearch_to_tsquery
 * (stemming, stop words, quoted phrases) and then turn the ANDs into ORs; ts_rank_cd still
 * ranks chunks that match more terms higher.
 */
export async function keywordSearch(db: Queryable, query: string, opts: SearchOptions = {}) {
  const { rows } = await db.query<RetrievedChunk>(
    `SELECT ${SELECT}, ts_rank_cd(c.tsv, s.q) AS score
       FROM chunks c JOIN documents d ON d.id = c.document_id,
            (SELECT replace(websearch_to_tsquery('english', $1)::text, ' & ', ' | ')::tsquery AS q) s
      WHERE s.q::text <> '' AND c.tsv @@ s.q AND ${DOC_FILTER}
      ORDER BY score DESC, c.id
      LIMIT $3`,
    [query, opts.documentIds?.length ? opts.documentIds : null, opts.limit ?? CANDIDATES_PER_RETRIEVER],
  );
  return rows.map((r) => ({ ...r, score: Number(r.score) }));
}

/**
 * Reciprocal Rank Fusion: score(d) = sum over retrievers of 1 / (k + rank(d)).
 * Uses only ranks, so cosine similarities and ts_rank scores never need to be put on one scale.
 */
export function reciprocalRankFusion(
  lists: { name: "vector" | "keyword"; results: RetrievedChunk[] }[],
  k = RRF_K,
): RetrievedChunk[] {
  const fused = new Map<number, RetrievedChunk>();
  for (const { name, results } of lists) {
    results.forEach((chunk, i) => {
      const rank = i + 1;
      const entry = fused.get(chunk.id) ?? { ...chunk, score: 0 };
      entry.score += 1 / (k + rank);
      if (name === "vector") entry.vectorRank = rank;
      else entry.keywordRank = rank;
      fused.set(chunk.id, entry);
    });
  }
  return [...fused.values()].sort((a, b) => b.score - a.score || a.id - b.id);
}

export async function retrieve(
  db: Queryable,
  embedder: Embedder,
  query: string,
  opts: SearchOptions & { mode?: RetrievalMode; topK?: number } = {},
): Promise<RetrievedChunk[]> {
  const mode = opts.mode ?? "hybrid";
  const topK = opts.topK ?? TOP_K;
  const searchOpts = { documentIds: opts.documentIds, limit: CANDIDATES_PER_RETRIEVER };

  if (mode === "keyword") return (await keywordSearch(db, query, searchOpts)).slice(0, topK);
  const embedding = await embedder.embedQuery(query);
  if (mode === "vector") return (await vectorSearch(db, embedding, searchOpts)).slice(0, topK);

  const [vector, keyword] = await Promise.all([
    vectorSearch(db, embedding, searchOpts),
    keywordSearch(db, query, searchOpts),
  ]);
  return reciprocalRankFusion([
    { name: "vector", results: vector },
    { name: "keyword", results: keyword },
  ]).slice(0, topK);
}
