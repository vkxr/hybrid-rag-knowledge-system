import { NextResponse } from "next/server";
import { z } from "zod";
import { chatEventStream, streamAnswer } from "@/lib/answer";
import { getDb } from "@/lib/db";
import { getEmbedder } from "@/lib/embeddings";
import { retrieve } from "@/lib/retrieval";

export const runtime = "nodejs";

const Body = z.object({
  question: z.string().trim().min(1).max(2000),
  documentIds: z.array(z.string().uuid()).max(50).optional(),
  mode: z.enum(["hybrid", "vector", "keyword"]).default("hybrid"),
});

export async function POST(req: Request) {
  const parsed = Body.safeParse(await req.json().catch(() => null));
  if (!parsed.success) {
    return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });
  }
  const { question, documentIds, mode } = parsed.data;
  const chunks = await retrieve(getDb(), getEmbedder(), question, { mode, documentIds });

  return new Response(chatEventStream(chunks, streamAnswer(question, chunks)), {
    headers: { "Content-Type": "application/x-ndjson; charset=utf-8", "Cache-Control": "no-cache" },
  });
}
