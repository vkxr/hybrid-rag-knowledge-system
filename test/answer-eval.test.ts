import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { chatEventStream, extractCitations, formatSources, streamAnswer } from "../lib/answer";
import type { Queryable } from "../lib/db";
import { ingestPages } from "../lib/ingest";
import type { Judge } from "../lib/judge";
import { firstRelevantRank, mapLimit, mrrAtK, percentile, recallAtK } from "../lib/metrics";
import type { RetrievedChunk } from "../lib/retrieval";
import { evaluateAnswers, evaluateRetrieval, type GoldenItem } from "../eval/evaluate";
import { FakeEmbedder, testDb } from "./helpers";

const chunk = (id: number, name = "a.pdf", page = 1): RetrievedChunk => ({
  id, documentId: "d", documentName: name, chunkIndex: id, page, content: `content ${id}`, score: 1,
});

describe("metrics", () => {
  it("computes rank, recall@k and MRR@k", () => {
    const retrieved = [chunk(4), chunk(7), chunk(9)];
    expect(firstRelevantRank(retrieved, [{ document: "a.pdf", chunkIndex: 7 }])).toBe(2);
    expect(firstRelevantRank(retrieved, [{ document: "b.pdf", chunkIndex: 7 }])).toBeNull();

    const ranks = [1, 2, null, 7];
    expect(recallAtK(ranks, 6)).toBe(0.5);
    expect(recallAtK(ranks, 1)).toBe(0.25);
    expect(mrrAtK(ranks, 6)).toBeCloseTo((1 + 0.5) / 4);
    expect(percentile([10, 20, 30], 0.5)).toBe(20);
  });

  it("mapLimit keeps order and respects the limit", async () => {
    let active = 0;
    let peak = 0;
    const out = await mapLimit([5, 1, 3, 2], 2, async (x) => {
      peak = Math.max(peak, ++active);
      await new Promise((r) => setTimeout(r, x));
      active--;
      return x * 10;
    });
    expect(out).toEqual([50, 10, 30, 20]);
    expect(peak).toBe(2);
  });
});

describe("answers and streaming", () => {
  it("formats numbered sources with document and page", () => {
    expect(formatSources([chunk(1, "hr.pdf", 4)])).toBe("[1] hr.pdf, page 4\ncontent 1");
  });

  it("extracts only valid, de-duplicated citations", () => {
    expect(extractCitations("Yes [2]. Also [1][2] and [9].", 3)).toEqual([1, 2]);
  });

  it("streams sources first, then tokens, then cited source numbers", async () => {
    async function* tokens() { yield "Refunds take 7 days "; yield "[1]."; }
    const text = await new Response(chatEventStream([chunk(1), chunk(2)], tokens())).text();
    const events = text.trim().split("\n").map((l) => JSON.parse(l));
    expect(events.map((e) => e.type)).toEqual(["sources", "token", "token", "done"]);
    expect(events[0].sources[0]).toMatchObject({ documentName: "a.pdf", page: 1, snippet: "content 1" });
    expect(events[0].sources[0].content).toBeUndefined();
    expect(events.at(-1).citations).toEqual([1]);
  });

  it("emits an error event if the model fails mid-stream", async () => {
    async function* tokens() { yield "partial"; throw new Error("boom"); }
    const text = await new Response(chatEventStream([chunk(1)], tokens())).text();
    expect(text.trim().split("\n").map((l) => JSON.parse(l).type)).toEqual(["sources", "token", "error"]);
  });

  it("streamAnswer yields text from the chat model stream", async () => {
    const model = {
      async stream() { return (async function* () { yield { content: "A" }; yield { content: "" }; yield { content: "B" }; })(); },
      async invoke() { return { content: "AB" }; },
    };
    const parts: string[] = [];
    for await (const p of streamAnswer("q", [chunk(1)], model)) parts.push(p);
    expect(parts).toEqual(["A", "B"]);
  });
});

describe("evaluation harness", () => {
  let db: Queryable & { close(): Promise<void> };
  beforeEach(async () => { db = await testDb(); });
  afterEach(async () => db.close());

  const golden: GoldenItem[] = [
    { id: "q1", question: "How long do refunds take to process?", referenceAnswer: "7 business days",
      relevant: [{ document: "policy.pdf", chunkIndex: 0 }] },
    { id: "q2", question: "Which city do orders ship from?", referenceAnswer: "Pune",
      relevant: [{ document: "policy.pdf", chunkIndex: 1 }] },
  ];

  it("scores vector, keyword and hybrid retrieval against the golden set", async () => {
    const embedder = new FakeEmbedder();
    await ingestPages(db, embedder, "policy.pdf", [
      "Refunds are processed within 7 business days to the original payment method.",
      "Orders ship from our Pune warehouse within 2 business days.",
    ]);
    const { summary, rows } = await evaluateRetrieval(db, embedder, golden, ["vector", "keyword", "hybrid"]);
    expect(summary.hybrid.recallAt6).toBe(1);
    expect(summary.hybrid.mrrAt6).toBe(1);
    expect(rows).toHaveLength(6);
  });

  it("aggregates judge scores for answers", async () => {
    const judge: Judge = {
      faithfulness: async (answer) => ({ score: answer.includes("7") ? 1 : 0.5, unsupported: [] }),
      correctness: async (_q, answer, ref) => ({ correct: answer.includes(ref), reasoning: "" }),
    };
    const { summary } = await evaluateAnswers(
      golden, "hybrid", async () => [chunk(0)],
      async (q) => (q.includes("refund") ? "7 business days [1]" : "Mumbai [1]"), judge,
    );
    expect(summary.faithfulness).toBe(0.75);
    expect(summary.answerAccuracy).toBe(0.5);
  });
});
