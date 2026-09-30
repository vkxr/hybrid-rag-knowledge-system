/** Bulk-ingest PDFs from a folder: npm run ingest -- ./corpus */
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { getDb } from "../lib/db";
import { getEmbedder } from "../lib/embeddings";
import { ingestPdf } from "../lib/ingest";

const dir = process.argv[2];
if (!dir) {
  console.error("Usage: npm run ingest -- <folder-with-pdfs>");
  process.exit(1);
}

const files = (await readdir(dir)).filter((f) => f.toLowerCase().endsWith(".pdf"));
for (const file of files) {
  const start = Date.now();
  try {
    const r = await ingestPdf(getDb(), getEmbedder(), file, new Uint8Array(await readFile(join(dir, file))));
    const note = r.duplicate ? "already indexed" : `${r.pages} pages, ${r.chunks} chunks`;
    console.log(`${file}: ${note} (${((Date.now() - start) / 1000).toFixed(1)}s)`);
  } catch (err) {
    console.error(`${file}: FAILED - ${err instanceof Error ? err.message : err}`);
  }
}
process.exit(0);
