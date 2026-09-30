import { formatSources } from "../lib/answer";
import { TOP_K } from "../lib/config";
import type { Queryable } from "../lib/db";
import type { Embedder } from "../lib/embeddings";
import type { Judge } from "../lib/judge";
import { firstRelevantRank, mapLimit, mean, mrrAtK, percentile, recallAtK } from "../lib/metrics";
import { type RetrievalMode, type RetrievedChunk, retrieve } from "../lib/retrieval";

export interface GoldenItem {
  id: string;
  question: string;
  referenceAnswer: string;
  relevant: { document: string; chunkIndex: number; page?: number }[];
  reviewed?: boolean;
}

export interface RetrievalRow {
  id: string;
  mode: RetrievalMode;
  rank: number | null;
  latencyMs: number;
  retrieved: string[];
}

export interface AnswerRow {
  id: string;
  mode: RetrievalMode;
  answer: string;
  faithfulness: number;
  unsupportedClaims: string[];
  correct: boolean;
  judgeReasoning: string;
}

export async function evaluateRetrieval(
  db: Queryable,
  embedder: Embedder,
  golden: GoldenItem[],
  modes: RetrievalMode[],
  concurrency = 4,
) {
  const rows: RetrievalRow[] = [];
  const summary: Record<string, { recallAt6: number; mrrAt6: number; recallAt1: number; p50LatencyMs: number }> = {};
  for (const mode of modes) {
    const modeRows = await mapLimit(golden, concurrency, async (item) => {
      const start = performance.now();
      const results = await retrieve(db, embedder, item.question, { mode, topK: TOP_K });
      return {
        id: item.id,
        mode,
        rank: firstRelevantRank(results, item.relevant),
        latencyMs: Math.round(performance.now() - start),
        retrieved: results.map((r) => `${r.documentName}#${r.chunkIndex}`),
      };
    });
    const ranks = modeRows.map((r) => r.rank);
    summary[mode] = {
      recallAt6: recallAtK(ranks, TOP_K),
      mrrAt6: mrrAtK(ranks, TOP_K),
      recallAt1: recallAtK(ranks, 1),
      p50LatencyMs: Math.round(percentile(modeRows.map((r) => r.latencyMs), 0.5)),
    };
    rows.push(...modeRows);
  }
  return { summary, rows };
}

export async function evaluateAnswers(
  golden: GoldenItem[],
  mode: RetrievalMode,
  retrieveFn: (q: string) => Promise<RetrievedChunk[]>,
  answerFn: (q: string, chunks: RetrievedChunk[]) => Promise<string>,
  judge: Judge,
  concurrency = 4,
) {
  const rows: AnswerRow[] = await mapLimit(golden, concurrency, async (item) => {
    const chunks = await retrieveFn(item.question);
    const answer = await answerFn(item.question, chunks);
    const [faith, corr] = await Promise.all([
      judge.faithfulness(answer, formatSources(chunks)),
      judge.correctness(item.question, answer, item.referenceAnswer),
    ]);
    return {
      id: item.id, mode, answer,
      faithfulness: faith.score, unsupportedClaims: faith.unsupported,
      correct: corr.correct, judgeReasoning: corr.reasoning,
    };
  });
  return {
    summary: {
      faithfulness: mean(rows.map((r) => r.faithfulness)),
      answerAccuracy: mean(rows.map((r) => (r.correct ? 1 : 0))),
    },
    rows,
  };
}
