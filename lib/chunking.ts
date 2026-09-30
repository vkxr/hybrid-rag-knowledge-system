import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { extractText } from "unpdf";
import { CHUNK_OVERLAP, CHUNK_SIZE } from "./config";

export interface Chunk {
  content: string;
  page: number; // 1-based
  chunkIndex: number; // 0-based position within the document
}

/** Extract text per page so every chunk can cite the page it came from. */
export async function extractPages(pdf: Uint8Array): Promise<string[]> {
  const { text } = await extractText(pdf, { mergePages: false });
  return text.map((page) => page.replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim());
}

export const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
});

/**
 * Chunk page by page. Chunks never span two pages, which keeps page citations exact.
 * Pages are short enough (usually 2-4k chars) that this costs little context.
 */
export async function chunkPages(pages: string[]): Promise<Chunk[]> {
  const chunks: Chunk[] = [];
  for (let i = 0; i < pages.length; i++) {
    if (!pages[i]) continue;
    for (const content of await splitter.splitText(pages[i])) {
      if (content.trim().length < 20) continue; // page numbers, stray headers
      chunks.push({ content, page: i + 1, chunkIndex: chunks.length });
    }
  }
  return chunks;
}
