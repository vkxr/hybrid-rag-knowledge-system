import { ChatOpenAI } from "@langchain/openai";
import type { BaseMessageLike } from "@langchain/core/messages";
import { CHAT_MODEL } from "./config";
import type { RetrievedChunk } from "./retrieval";

export interface ChatModel {
  stream(messages: BaseMessageLike[]): Promise<AsyncIterable<{ content: unknown }>>;
  invoke(messages: BaseMessageLike[]): Promise<{ content: unknown }>;
}

let chat: ChatModel | undefined;

export function getChatModel(): ChatModel {
  chat ??= new ChatOpenAI({ model: CHAT_MODEL, temperature: 0 });
  return chat;
}

export const SYSTEM_PROMPT = `You answer questions using only the numbered sources provided.
- Cite every factual claim with the source number in square brackets, e.g. [2] or [1][3].
- If the sources do not contain the answer, say you could not find it in the documents. Do not guess.
- Be concise. Quote numbers, dates and names exactly as they appear in the sources.`;

export function formatSources(chunks: RetrievedChunk[]): string {
  return chunks
    .map((c, i) => `[${i + 1}] ${c.documentName}, page ${c.page}\n${c.content}`)
    .join("\n\n---\n\n");
}

export function buildMessages(question: string, chunks: RetrievedChunk[]): BaseMessageLike[] {
  return [
    ["system", SYSTEM_PROMPT],
    ["human", `Sources:\n\n${formatSources(chunks)}\n\nQuestion: ${question}`],
  ];
}

/** Source numbers the answer actually cited, e.g. "[1][3]" -> [1, 3]. Ignores out-of-range numbers. */
export function extractCitations(answer: string, sourceCount: number): number[] {
  const found = new Set<number>();
  for (const m of answer.matchAll(/\[(\d+)\]/g)) {
    const n = Number(m[1]);
    if (n >= 1 && n <= sourceCount) found.add(n);
  }
  return [...found].sort((a, b) => a - b);
}

function text(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map((p) => (typeof p === "string" ? p : p?.text ?? "")).join("");
  return "";
}

export async function* streamAnswer(question: string, chunks: RetrievedChunk[], model = getChatModel()) {
  for await (const part of await model.stream(buildMessages(question, chunks))) {
    const t = text(part.content);
    if (t) yield t;
  }
}

export async function generateAnswer(question: string, chunks: RetrievedChunk[], model = getChatModel()) {
  return text((await model.invoke(buildMessages(question, chunks))).content);
}

export type Source = Omit<RetrievedChunk, "content"> & { snippet: string };

export type ChatEvent =
  | { type: "sources"; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "done"; citations: number[] }
  | { type: "error"; message: string };

/** Newline-delimited JSON stream: sources first (so citations render immediately), then tokens. */
export function chatEventStream(
  chunks: RetrievedChunk[],
  tokens: AsyncIterable<string>,
): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  const line = (e: unknown) => enc.encode(`${JSON.stringify(e)}\n`);
  return new ReadableStream({
    async start(controller) {
      controller.enqueue(line({
        type: "sources",
        sources: chunks.map(({ content, ...rest }) => ({ ...rest, snippet: content.slice(0, 280) })),
      }));
      let answer = "";
      try {
        for await (const t of tokens) {
          answer += t;
          controller.enqueue(line({ type: "token", text: t }));
        }
        controller.enqueue(line({ type: "done", citations: extractCitations(answer, chunks.length) }));
      } catch (err) {
        console.error("answer stream failed", err);
        controller.enqueue(line({ type: "error", message: "The model call failed. Try again." }));
      }
      controller.close();
    },
  });
}
