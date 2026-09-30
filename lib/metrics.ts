/** Retrieval and answer-quality metrics. Pure functions, unit tested. */

/**
 * Rank (1-based) of the first retrieved chunk that is relevant, or null if none is.
 * A golden item is identified by document name + chunk index, which survive re-ingestion
 * (database ids don't).
 */
export function firstRelevantRank(
  retrieved: { documentName: string; chunkIndex: number }[],
  relevant: { document: string; chunkIndex: number }[],
): number | null {
  const want = new Set(relevant.map((r) => `${r.document}#${r.chunkIndex}`));
  const i = retrieved.findIndex((c) => want.has(`${c.documentName}#${c.chunkIndex}`));
  return i === -1 ? null : i + 1;
}

/** Fraction of questions whose relevant chunk appears in the top k. */
export function recallAtK(ranks: (number | null)[], k: number): number {
  if (ranks.length === 0) return 0;
  return ranks.filter((r) => r !== null && r <= k).length / ranks.length;
}

/** Mean reciprocal rank, counting only hits within the top k. */
export function mrrAtK(ranks: (number | null)[], k: number): number {
  if (ranks.length === 0) return 0;
  return ranks.reduce<number>((sum, r) => sum + (r !== null && r <= k ? 1 / r : 0), 0) / ranks.length;
}

export function mean(values: number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

export function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const k = (s.length - 1) * p;
  const lo = Math.floor(k);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo] + (s[hi] - s[lo]) * (k - lo);
}

/** Run async work over items with bounded concurrency, preserving order. */
export async function mapLimit<T, R>(items: T[], limit: number, fn: (item: T, i: number) => Promise<R>) {
  const out = new Array<R>(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i], i);
    }
  });
  await Promise.all(workers);
  return out;
}
