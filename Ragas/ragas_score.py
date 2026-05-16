import json
from ragas import evaluate,EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# ── Load eval results from previous step ─────────────────
with open("eval_results.json", "r") as f:
    eval_results = json.load(f)

print(f"Loaded {len(eval_results)} eval results")
print()

# ── Convert to RAGAS Dataset format ──────────────────────
# RAGAS needs specific field names — map your results to them
samples=[]
for r in eval_results:
    sample=SingleTurnSample(
        user_input=r["question"],
        response=r["answer"],
        reference=r["ground_truth"],
        retrieved_contexts=r["contexts"]
    )
    samples.append(sample)

dataset=EvaluationDataset(samples=samples)
print("Dataset created:")
print(f"  Rows: {len(dataset)}")
print()

# ── Configure RAGAS to use Ollama ─────────────────────────
# RAGAS uses LLM-as-judge — it needs an LLM to score your answers
# We wrap Ollama so RAGAS can use it
# ollama_llm = OpenAI(
#     base_url="http://localhost:11434/v1", 
#     api_key="ollama" 
# # )

# ragas_embeddings=embedding_factory("openai",
#     model="mxbai-embed-large", 
#     client=ollama_llm)

# ragas_llm=llm_factory(model="llama3.1",provider="openai",client=ollama_llm)
ollama_llm = ChatOllama(model="llama3.1", temperature=0)
ollama_embeddings = OllamaEmbeddings(model="mxbai-embed-large")

ragas_llm = LangchainLLMWrapper(ollama_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)

# ── Run RAGAS evaluation ──────────────────────────────────
print("Running RAGAS evaluation...")
print("This takes 2-5 minutes — RAGAS calls LLM for each metric per question")
print()

faithfulness        = Faithfulness(llm=ragas_llm)
answer_relevancy    = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
context_precision   = ContextPrecision(llm=ragas_llm)
context_recall      = ContextRecall(llm=ragas_llm)

# Setting max_workers to 1 or 2 to stop overwhelming Ollama
# You can also explicitly increase the RAGAS timeout here
custom_config = RunConfig(max_workers=2, timeout=180)

results=evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    run_config=custom_config
)

# ── Display scores ────────────────────────────────────────
print("=" * 60)
print("RAGAS EVALUATION RESULTS")
print("=" * 60)

scores=results.to_pandas()

print(f"\nOverall Scores:")
print(f"  Faithfulness:     {scores['faithfulness'].mean():.4f}")
print(f"  Answer Relevancy: {scores['answer_relevancy'].mean():.4f}")
print(f"  Context Precision:{scores['context_precision'].mean():.4f}")
print(f"  Context Recall:   {scores['context_recall'].mean():.4f}")

# ── Find worst performing questions ──────────────────────
print(f"\nPer-question breakdown:")
print("-" * 60)

for i ,row in scores.iterrows():
    q=eval_results[i]["question"][:50]
    print(f"Q{i+1}:{q}...")
    print(f"  Faith: {row['faithfulness']:.2f} | "
          f"Rel: {row['answer_relevancy']:.2f} | "
          f"Prec: {row['context_precision']:.2f} | "
          f"Rec: {row['context_recall']:.2f}")
    
# ── Identify weakest metric ───────────────────────────────
print("\n" + "=" * 60)
print("DIAGNOSIS")
print("=" * 60)    

avg_score={
    "faithfulness":scores['faithfulness'].mean(),
    "answer_relevancy": scores['answer_relevancy'].mean(),
    "context_precision": scores['context_precision'].mean(),
    "context_recall": scores['context_recall'].mean(),
}

weakest=min(avg_score,key=avg_score.get)
print(f"Weakest metric: {weakest} ({avg_score[weakest]:.4f})")

diagnosis = {
    "faithfulness": "LLM is hallucinating — answers not grounded in context. Fix: make RAG prompt stricter. Add 'ONLY use the context provided.'",
    "answer_relevancy": "Answers are off-topic or incomplete. Fix: improve prompt to be more direct. Check if retrieval is returning wrong docs.",
    "context_precision": "Wrong chunks being retrieved. Fix: try smaller chunk size, better metadata filtering, or hybrid search.",
    "context_recall": "Missing relevant chunks. Fix: increase k (retrieve more docs), or check if relevant docs are even in your vector store."
}

print(f"What it means: {diagnosis[weakest]}")

#save score to file
final_score={
    "faithfulness":float(avg_score["faithfulness"]),
    "answer_relevancy": float(avg_score["answer_relevancy"]),
    "context_precision": float(avg_score["context_precision"]),
    "context_recall": float(avg_score["context_recall"]),
    "weakest_metric": weakest
}

with open("./ragas_scores.json", "w") as f:
    json.dump(final_score, f, indent=2)

print(f"\nScores saved to rag/ragas_scores.json")
print("Add these numbers to your GitHub README — this is your proof of work")    




