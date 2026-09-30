import { OpenAIEmbeddings } from "@langchain/openai";
import { EMBEDDING_DIM, EMBEDDING_MODEL } from "./config";

export interface Embedder {
  embedDocuments(texts: string[]): Promise<number[][]>;
  embedQuery(text: string): Promise<number[]>;
}

let embedder: Embedder | undefined;

export function getEmbedder(): Embedder {
  embedder ??= new OpenAIEmbeddings({
    model: EMBEDDING_MODEL,
    dimensions: EMBEDDING_DIM,
    batchSize: 100,
  });
  return embedder;
}
