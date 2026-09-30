/**
 * Run the evaluation.
 *
 *   npm run eval                    retrieval metrics for vector, keyword, hybrid + LLM-judged answers (hybrid)
 *   npm run eval -- --no-judge      retrieval metrics only (no chat-model cost)
 *   npm run eval -- --judge-all     judge answers for all three modes
 *
 * Writes eval/results/<timestamp>.json and .md
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { generateAnswer } from "../lib/answer";
import { CHAT_MODEL, EMBEDDING_MODEL, TOP_K } from "../lib/config";
import { getDb } from "../lib/db";
import { getEmbedder } from "../lib/embeddings";
import { createJudge } from "../lib/judge";
import { type RetrievalMode, retrieve } from "../lib/retrieval";
import { evaluateAnswers, evaluateRetrieval, type GoldenItem } from "./evaluate";

const args = new Set(process.argv.slice(2));
const MODES: RetrievalMode[] = ["vector", "keyword", "hybrid"];
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

const golden: GoldenItem[] = JSON.parse(await readFile(new URL("./golden.json", import.meta.url), "utf8"));
const unreviewed = golden.filter((g) => !g.reviewed).length;
if (unreviewed) console.warn(`Warning: ${unreviewed}/${golden.length} golden questions not marked reviewed.`);

const db = getDb();
const embedder = getEmbedder();

console.log(`Evaluating retrieval on ${golden.length} questions...`);
const retrieval = await evaluateRetrieval(db, embedder, golden, MODES);

const answers: Record<string, Awaited<ReturnType<typeof evaluateAnswers>>> = {};
if (!args.has("--no-judge")) {
  const judge = createJudge();
  for (const mode of args.has("--judge-all") ? MODES : (["hybrid"] as RetrievalMode[])) {
    console.log(`Generating and judging answers (${mode})...`);
    answers[mode] = await evaluateAnswers(
      golden, mode, (q) => retrieve(db, embedder, q, { mode }), (q, c) => generateAnswer(q, c), judge,
    );
  }
}

const lines = [
  `# RAG evaluation`,
  "",
  `${golden.length} questions (${golden.length - unreviewed} hand-reviewed). Embeddings: ${EMBEDDING_MODEL}. ` +
    `Answers and judge: ${CHAT_MODEL}. Top-k: ${TOP_K}.`,
  "",
  "| Retrieval | Recall@6 | MRR@6 | Recall@1 | p50 latency | Faithfulness | Answer accuracy |",
  "|---|---|---|---|---|---|---|",
  ...MODES.map((m) => {
    const r = retrieval.summary[m];
    const a = answers[m]?.summary;
    return `| ${m} | ${pct(r.recallAt6)} | ${r.mrrAt6.toFixed(3)} | ${pct(r.recallAt1)} | ${r.p50LatencyMs} ms | ` +
      `${a ? pct(a.faithfulness) : "-"} | ${a ? pct(a.answerAccuracy) : "-"} |`;
  }),
  "",
  "- Recall@6: share of questions whose source chunk is in the 6 chunks sent to the model.",
  "- MRR@6: mean of 1/rank of the source chunk (0 if not in the top 6).",
  "- Faithfulness: share of the answer's factual claims supported by the retrieved sources (LLM judge).",
  "- Answer accuracy: share of answers containing the reference answer's key facts (LLM judge).",
];
const report = `${lines.join("\n")}\n`;

const dir = new URL("./results/", import.meta.url);
await mkdir(dir, { recursive: true });
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
await writeFile(new URL(`${stamp}.json`, dir), JSON.stringify({ retrieval, answers }, null, 2));
await writeFile(new URL(`${stamp}.md`, dir), report);
console.log(`\n${report}\nSaved eval/results/${stamp}.json and .md`);
process.exit(0);
