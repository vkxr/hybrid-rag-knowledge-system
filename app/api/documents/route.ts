import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const { rows } = await getDb().query(
    `SELECT d.id, d.name, d.page_count AS "pageCount", d.created_at AS "createdAt",
            count(c.id)::int AS "chunkCount"
       FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
      GROUP BY d.id ORDER BY d.created_at DESC`,
  );
  return NextResponse.json({ documents: rows });
}
