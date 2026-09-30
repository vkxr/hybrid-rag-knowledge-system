import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { CHAT_MODEL } from "./config";

/** LLM-as-judge scorers. Each returns a score plus the judge's reasoning, for spot-checking. */

const FaithfulnessSchema = z.object({
  claims: z.array(z.object({
    claim: z.string().describe("One factual claim made in the answer"),
    supported: z.boolean().describe("True only if the sources state or directly imply this claim"),
  })).describe("Every factual claim in the answer. Empty if the answer makes none (e.g. 'not found')."),
});

const CorrectnessSchema = z.object({
  reasoning: z.string().describe("Compare the answer to the reference in one or two sentences"),
  correct: z.boolean().describe("True if the answer contains the key facts of the reference answer"),
});

export interface Judge {
  faithfulness(answer: string, sources: string): Promise<{ score: number; unsupported: string[] }>;
  correctness(question: string, answer: string, reference: string): Promise<{ correct: boolean; reasoning: string }>;
}

export function createJudge(model = process.env.JUDGE_MODEL ?? CHAT_MODEL): Judge {
  const llm = new ChatOpenAI({ model, temperature: 0 });
  const faith = llm.withStructuredOutput(FaithfulnessSchema, { name: "faithfulness" });
  const correct = llm.withStructuredOutput(CorrectnessSchema, { name: "correctness" });

  return {
    async faithfulness(answer, sources) {
      const { claims } = await faith.invoke([
        ["system", "You check whether an answer is grounded in its sources. List each factual claim " +
          "in the answer and mark whether the sources support it. Ignore citation markers like [1]."],
        ["human", `Sources:\n${sources}\n\nAnswer:\n${answer}`],
      ]);
      // An answer with no factual claims ("not found in the documents") is trivially faithful.
      const score = claims.length ? claims.filter((c) => c.supported).length / claims.length : 1;
      return { score, unsupported: claims.filter((c) => !c.supported).map((c) => c.claim) };
    },
    async correctness(question, answer, reference) {
      return correct.invoke([
        ["system", "You grade answers against a reference answer. Wording may differ; facts must match."],
        ["human", `Question: ${question}\n\nReference answer: ${reference}\n\nAnswer to grade: ${answer}`],
      ]);
    },
  };
}
