import { NextResponse } from "next/server";
import { MAX_UPLOAD_BYTES } from "@/lib/config";
import { getDb } from "@/lib/db";
import { getEmbedder } from "@/lib/embeddings";
import { ingestPdf } from "@/lib/ingest";

export const runtime = "nodejs";
export const maxDuration = 300;

export async function POST(req: Request) {
  const form = await req.formData().catch(() => null);
  const files = form?.getAll("files").filter((f): f is File => f instanceof File) ?? [];
  if (files.length === 0) {
    return NextResponse.json({ error: "Attach one or more PDFs in the 'files' field" }, { status: 400 });
  }

  const results = [];
  for (const file of files) {
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      results.push({ name: file.name, error: "Not a PDF" });
      continue;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      results.push({ name: file.name, error: "File is larger than 20 MB" });
      continue;
    }
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      results.push(await ingestPdf(getDb(), getEmbedder(), file.name, bytes));
    } catch (err) {
      console.error("ingest failed", file.name, err);
      results.push({ name: file.name, error: err instanceof Error ? err.message : "Ingestion failed" });
    }
  }
  const status = results.every((r) => "error" in r) ? 422 : 200;
  return NextResponse.json({ results }, { status });
}
