import json
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas import evaluate,EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

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

results=evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
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



