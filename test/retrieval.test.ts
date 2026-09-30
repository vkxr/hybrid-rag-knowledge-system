import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { chunkPages } from "../lib/chunking";
import { CHUNK_OVERLAP, CHUNK_SIZE } from "../lib/config";
import type { Queryable } from "../lib/db";
import { ingestPages } from "../lib/ingest";
import { keywordSearch, reciprocalRankFusion, retrieve, type RetrievedChunk } from "../lib/retrieval";
import { FakeEmbedder, testDb } from "./helpers";

const REFUNDS = [
  "Refund policy. Customers may request a full refund within 30 days of delivery. Refunds are issued to the original payment method within 7 business days.",
  "Shipping. Orders ship from Pune within 2 business days. Express delivery is available in 40 cities.",
];
const HR = [
  "Leave policy. Employees receive 24 days of paid annual leave, accrued monthly.",
  "Parental leave. Primary caregivers receive 26 weeks of paid parental leave as per the Maternity Benefit Act.",
];

let db: Queryable & { close(): Promise<void> };
let embedder: FakeEmbedder;

beforeEach(async () => {
  db = await testDb();
  embedder = new FakeEmbedder();
});
afterEach(async () => db.close());

describe("chunking", () => {
  it("uses 1,800-character chunks with 200-character overlap and keeps page numbers", async () => {
    const words = Array.from({ length: 900 }, (_, i) => `word${i}`).join(" ");
    const chunks = await chunkPages([words, "short second page with enough text"]);
    const firstPage = chunks.filter((c) => c.page === 1);
    expect(firstPage.length).toBeGreaterThan(1);
    expect(Math.max(...firstPage.map((c) => c.content.length))).toBeLessThanOrEqual(CHUNK_SIZE);
    // consecutive chunks overlap
    const tail = firstPage[0].content.slice(-CHUNK_OVERLAP / 2);
    expect(firstPage[1].content).toContain(tail.trim().split(" ").at(-1));
    expect(chunks.at(-1)?.page).toBe(2);
    expect(chunks.map((c) => c.chunkIndex)).toEqual(chunks.map((_, i) => i));
  });
});

describe("ingestion", () => {
  it("stores chunks with 1,536-dim embeddings and deduplicates re-uploads", async () => {
    const first = await ingestPages(db, embedder, "refunds.pdf", REFUNDS);
    expect(first).toMatchObject({ chunks: 2, pages: 2, duplicate: false });
    const dims = await db.query<{ d: number }>("SELECT vector_dims(embedding) AS d FROM chunks LIMIT 1");
    expect(dims.rows[0].d).toBe(1536);

    const again = await ingestPages(db, embedder, "refunds-copy.pdf", REFUNDS);
    expect(again).toMatchObject({ documentId: first.documentId, duplicate: true, chunks: 2 });
    const count = await db.query<{ n: number }>("SELECT count(*)::int AS n FROM documents");
    expect(count.rows[0].n).toBe(1);
  });

  it("rolls back the document if inserting chunks fails", async () => {
    const broken = { ...embedder, embedDocuments: async (t: string[]) => t.map(() => [1, 2, 3]) };
    await expect(ingestPages(db, broken as FakeEmbedder, "bad.pdf", REFUNDS)).rejects.toThrow();
    const count = await db.query<{ n: number }>("SELECT count(*)::int AS n FROM documents");
    expect(count.rows[0].n).toBe(0);
  });
});

describe("retrieval", () => {
  it("vector, keyword and hybrid modes find the right chunk with page metadata", async () => {
    await ingestPages(db, embedder, "refunds.pdf", REFUNDS);
    await ingestPages(db, embedder, "hr.pdf", HR);
    for (const mode of ["vector", "keyword", "hybrid"] as const) {
      const [top] = await retrieve(db, embedder, "how many days for a refund", { mode });
      expect(top.documentName, mode).toBe("refunds.pdf");
      expect(top.page, mode).toBe(1);
    }
  });

  it("hybrid results carry ranks from both retrievers and respect top-k", async () => {
    await ingestPages(db, embedder, "refunds.pdf", REFUNDS);
    await ingestPages(db, embedder, "hr.pdf", HR);
    const results = await retrieve(db, embedder, "paid parental leave weeks", { topK: 3 });
    expect(results.length).toBeLessThanOrEqual(3);
    expect(results[0].content).toContain("Parental leave");
    expect(results[0].vectorRank).toBe(1);
    expect(results[0].keywordRank).toBe(1);
  });

  it("multi-document search can be restricted to selected documents", async () => {
    await ingestPages(db, embedder, "refunds.pdf", REFUNDS);
    const hr = await ingestPages(db, embedder, "hr.pdf", HR);
    const results = await retrieve(db, embedder, "refund within 30 days", { documentIds: [hr.documentId] });
    expect(results.every((r) => r.documentName === "hr.pdf")).toBe(true);
  });

  it("keyword search handles queries with punctuation, stop words only, and no matches", async () => {
    await ingestPages(db, embedder, "refunds.pdf", REFUNDS);
    expect(await keywordSearch(db, "what's the 'refund' policy?!")).not.toHaveLength(0);
    expect(await keywordSearch(db, "kubernetes")).toHaveLength(0);
    expect(await keywordSearch(db, "the and of")).toHaveLength(0);
  });

  it("keyword search keeps quoted phrases as phrases", async () => {
    await ingestPages(db, embedder, "refunds.pdf", REFUNDS);
    const [top] = await keywordSearch(db, '"express delivery"');
    expect(top.content).toContain("Express delivery");
    expect(await keywordSearch(db, '"delivery express"')).toHaveLength(0);
  });
});

describe("reciprocal rank fusion", () => {
  const c = (id: number) => ({ id }) as RetrievedChunk;

  it("rewards chunks ranked well by both retrievers", () => {
    const fused = reciprocalRankFusion([
      { name: "vector", results: [c(1), c(2), c(3)] },
      { name: "keyword", results: [c(3), c(2), c(9)] },
    ]);
    // 2: 1/62 + 1/62, 3: 1/63 + 1/61, 1: 1/61 only
    expect(fused.map((r) => r.id)).toEqual([3, 2, 1, 9]);
    expect(fused[0]).toMatchObject({ vectorRank: 3, keywordRank: 1 });
    expect(fused[0].score).toBeCloseTo(1 / 63 + 1 / 61, 10);
  });
});
