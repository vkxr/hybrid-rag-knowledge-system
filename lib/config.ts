/** Tunable RAG settings. The values match the project's documented configuration. */
export const CHUNK_SIZE = 1800; // characters
export const CHUNK_OVERLAP = 200; // characters
export const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL ?? "text-embedding-3-small";
export const EMBEDDING_DIM = 1536;
export const CHAT_MODEL = process.env.CHAT_MODEL ?? "gpt-4o-mini";
export const TOP_K = 6; // chunks sent to the model
export const CANDIDATES_PER_RETRIEVER = 20; // candidates each retriever contributes before fusion
export const RRF_K = 60; // standard Reciprocal Rank Fusion constant
export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
