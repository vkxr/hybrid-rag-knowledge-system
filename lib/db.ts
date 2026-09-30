import pg from "pg";
import { EMBEDDING_DIM } from "./config";

/** Anything with a pg-style `query` method: a pg Pool in the app, PGlite in tests. */
export interface Queryable {
  query<T = any>(sql: string, params?: unknown[]): Promise<{ rows: T[] }>;
}

export const SCHEMA = `
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name        text NOT NULL,
  sha256      text NOT NULL UNIQUE,
  page_count  integer NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id           bigserial PRIMARY KEY,
  document_id  uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index  integer NOT NULL,
  page         integer NOT NULL,
  content      text NOT NULL,
  embedding    vector(${EMBEDDING_DIM}) NOT NULL,
  tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_document_id ON chunks (document_id);
`;

export async function migrate(db: Queryable & { exec?: (sql: string) => Promise<unknown> }) {
  if (db.exec) {
    await db.exec(SCHEMA); // PGlite: multi-statement exec
  } else {
    await db.query(SCHEMA); // pg: simple query protocol accepts multiple statements
  }
}

export function toVectorLiteral(values: number[]): string {
  return `[${values.join(",")}]`;
}

let pool: pg.Pool | undefined;

export function getDb(): Queryable {
  if (!pool) {
    if (!process.env.DATABASE_URL) throw new Error("DATABASE_URL is not set");
    pool = new pg.Pool({ connectionString: process.env.DATABASE_URL, max: 10 });
  }
  return pool;
}
