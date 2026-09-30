/**
 * Build the golden evaluation set from your indexed documents.
 *
 *   npm run eval:golden -- --n 100
 *
 * Samples chunks evenly across documents and asks the model to write one question that the chunk
 * answers, plus a reference answer. Writes eval/golden.json.
 *
 * IMPORTANT: review the file by hand before trusting results. Delete or fix questions that are
 * vague, answerable from general knowledge, or answered equally well by several chunks.
 * Set "reviewed": true on each item you have checked.
 */
import { writeFile } from "node:fs/promises";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { CHAT_MODEL } from "../lib/config";
import { getDb } from "../lib/db";
import { mapLimit } from "../lib/metrics";

const n = Number(process.argv[process.argv.indexOf("--n") + 1]) || 100;
const OUT = new URL("./golden.json", import.meta.url);

const QA = z.object({
  question: z.string().describe(
    "A specific question a user of these documents would realistically ask, answerable from this " +
    "passage alone. Do not mention 'the passage' or 'the document'. Prefer questions about specific " +
    "facts, numbers, conditions or procedures."),
  answer: z.string().describe("The correct answer, in one or two sentences, using only the passage"),
  usable: z.boolean().describe("False if the passage is boilerplate (table of contents, headers, legal filler)"),
});

// Oversample, then drop passages the model marks as unusable.
const { rows } = await getDb().query<{ document: string; chunkIndex: number; page: number; content: string }>(
  `SELECT document, "chunkIndex", page, content FROM (
     SELECT d.name AS document, c.chunk_index AS "chunkIndex", c.page, c.content,
            row_number() OVER (PARTITION BY c.document_id ORDER BY random()) AS rn
       FROM chunks c JOIN documents d ON d.id = c.document_id
      WHERE length(c.content) > 500
   ) s
   ORDER BY rn, random()
   LIMIT $1`,
  [Math.ceil(n * 1.4)],
);
if (rows.length === 0) {
  console.error("No chunks found. Ingest some PDFs first (npm run ingest -- ./corpus).");
  process.exit(1);
}

const writer = new ChatOpenAI({ model: CHAT_MODEL, temperature: 0.3 }).withStructuredOutput(QA, { name: "qa" });
const generated = await mapLimit(rows, 8, async (chunk) => {
  const qa = await writer.invoke([
    ["system", "You write evaluation questions for a document search system."],
    ["human", `Passage from "${chunk.document}", page ${chunk.page}:\n\n${chunk.content}`],
  ]);
  return { chunk, qa };
});

const golden = generated
  .filter((g) => g.qa.usable)
  .slice(0, n)
  .map((g, i) => ({
    id: `q${String(i + 1).padStart(3, "0")}`,
    question: g.qa.question,
    referenceAnswer: g.qa.answer,
    relevant: [{ document: g.chunk.document, chunkIndex: g.chunk.chunkIndex, page: g.chunk.page }],
    reviewed: false,
  }));

await writeFile(OUT, `${JSON.stringify(golden, null, 2)}\n`);
console.log(`Wrote ${golden.length} questions to eval/golden.json. Review them before running the eval.`);
process.exit(0);
