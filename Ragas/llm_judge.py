from langchain_ollama import ChatOllama
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm= ChatOllama(model="llama3.1",temperature=0)

#llm judge promot
JUDGE_PROMPT=ChatPromptTemplate.from_template("""
You are an expert evaluator of AI-generated answers.
Score the answer on these 3 criteria. Return ONLY a JSON object.

Question: {question}
Context provided to the AI: {context}
AI Answer: {answer}
Ground Truth (correct answer): {ground_truth}

Score each criterion from 0.0 to 1.0:

1. faithfulness: Is every claim in the answer supported by the context?
   1.0 = fully grounded, 0.5 = partially grounded, 0.0 = hallucinated

2. relevancy: Does the answer directly address the question?
   1.0 = fully relevant, 0.5 = partially relevant, 0.0 = off-topic

3. correctness: How close is the answer to the ground truth?
   1.0 = identical meaning, 0.5 = partially correct, 0.0 = wrong

Return ONLY this JSON with no explanation:
{{
  "faithfulness": ,
  "relevancy": ,
  "correctness": ,
  "reasoning": ""
}}

""")

judge_chain = (
    JUDGE_PROMPT
    | llm
    | StrOutputParser()
)
def judge_answer(question:str,context:str,answer:str,ground_truth:str) -> dict:
    raw=judge_chain.invoke({"question": question,
        "context": context,
        "answer": answer,
        "ground_truth": ground_truth})
    #clean and parse json
    raw=raw.strip()
    if raw.startswith("'''"):
        raw=raw.split("'''")[1]
        if raw.startswith("json"):
            raw=raw[4:]
    raw=raw.strip()   

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return{
            "faithfulness": 0.0,
            "relevancy": 0.0,
            "correctness": 0.0,
            "reasoning": f"Parse error: {raw[:100]}"
        }
# ── Load eval results from step 5b ────────────────────────
with open("./rag/eval_results.json", "r") as f:
    eval_results = json.load(f)

# ── Score each Q&A pair ───────────────────────────────────
print("Running LLM-as-Judge evaluation...")
print("=" * 60)

all_scores = []

for i ,item in enumerate(eval_results):
    context="\n".join(item["contexts"])
    scores=judge_answer(
        question=item["question"],
        context=context,
        answer=item["answer"],
        ground_truth=item["ground_truth"]
    )

    all_scores.append(scores)

    print(f"Q{i+1}: {item['question'][:55]}...")
    print(f"  Faithfulness: {scores.get('faithfulness', 0):.2f} | "
          f"Relevancy: {scores.get('relevancy', 0):.2f} | "
          f"Correctness: {scores.get('correctness', 0):.2f}")
    print(f"  Reasoning: {scores.get('reasoning', '')[:80]}...")
    print()


# ── Summary ───────────────────────────────────────────────
print("=" * 60)
print("LLM-AS-JUDGE SUMMARY")
print("=" * 60)

avg_faith=sum(s.get("faithfulness",0) for s in all_scores)/len(all_scores)
avg_rel=sum(s.get("relevancy",0) for s in all_scores)/len(all_scores)
avg_corr=sum(s.get("correctness",0) for s in all_scores)/len(all_scores)

print(f"Average Faithfulness: {avg_faith:.4f}")
print(f"Average Relevancy:    {avg_rel:.4f}")
print(f"Average Correctness:  {avg_corr:.4f}")

# ── Compare with RAGAS ────────────────────────────────────
print()
try:
    with open("./rag/ragas_scores.json") as f:
        ragas = json.load(f)
    print("Comparison with RAGAS:")
    print(f"  Faithfulness — RAGAS: {ragas['faithfulness']:.4f} | "
          f"LLM-Judge: {avg_faith:.4f}")
    print(f"  Relevancy    — RAGAS: {ragas['answer_relevancy']:.4f} | "
          f"LLM-Judge: {avg_rel:.4f}")
    print()
       
except FileNotFoundError :
       print("Run ragas_score.py first to compare")


# ── Save judge results ────────────────────────────────────
output = {
    "avg_faithfulness": avg_faith,
    "avg_relevancy": avg_rel,
    "avg_correctness": avg_corr,
    "per_question": all_scores
}

with open("./rag/llm_judge_scores.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved to rag/llm_judge_scores.json")

 
