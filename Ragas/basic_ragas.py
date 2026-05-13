from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import json

embedding=OllamaEmbeddings(model="mxbai-embed-large")

llm=ChatOllama(
    model="llama3.1",
    temperature=0
)

PERSIST_DIR = "../Vector_Database/chroma_db"
COLLECTION_NAME = "ai_knowledge_base"

vector_db=Chroma(
    embedding_function=embedding,
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME
)

retrival_docs=vector_db.as_retriever(search_kwargs={"k":3})

RAG_PROMPT = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


rag_chain=({"context":retrival_docs|format_docs,"question":RunnablePassthrough()}
           |RAG_PROMPT
           |llm
           |StrOutputParser()
           )

# ── EVAL DATASET ──────────────────────────────────────────
# 10 Q&A pairs — questions your RAG should be able to answer
# ground_truth = the correct answer based on your documents
eval_questions = [
    {
        "question": "What is LangGraph and what does it use to control flow?",
        "ground_truth": "LangGraph is a library for building stateful multi-actor applications. It uses a state machine where nodes are functions and edges control flow between them."
    },
    {
        "question": "What does RAG stand for and what is its purpose?",
        "ground_truth": "RAG stands for Retrieval Augmented Generation. It retrieves relevant documents from a vector store and uses them as context for the LLM to generate grounded answers."
    },
    {
        "question": "What is ChromaDB and how can it run?",
        "ground_truth": "ChromaDB is an open-source vector database that stores embeddings and allows fast similarity search. It can run in-memory or persist to disk."
    },
    {
        "question": "What is CrewAI used for?",
        "ground_truth": "CrewAI is a framework for orchestrating role-playing autonomous AI agents. Agents have roles, goals, and backstories and work together to complete complex tasks."
    },
    {
        "question": "What programming language is Qdrant written in?",
        "ground_truth": "Qdrant is written in Rust and offers high performance similarity search with rich filtering on payload metadata."
    },
    {
        "question": "What metrics does RAGAS measure?",
        "ground_truth": "RAGAS measures faithfulness, answer relevancy, context precision, and context recall using LLM-as-judge."
    },
    {
        "question": "What techniques does prompt engineering include?",
        "ground_truth": "Prompt engineering techniques include chain-of-thought, few-shot examples, and ReAct patterns."
    },
    {
        "question": "How does function calling work with LLMs?",
        "ground_truth": "Function calling allows LLMs to request execution of predefined functions. The LLM returns structured JSON with the function name and arguments, and your code runs the function."
    },
    {
        "question": "Which vector database is good for production RAG systems?",
        "ground_truth": "Qdrant is good for production RAG systems as it offers high performance similarity search with rich filtering on payload metadata."
    },
    {
        "question": "What is the capital of France?",
        "ground_truth": "This information is not available in the knowledge base."
    },
]

# ── GENERATE: run each question through your RAG pipeline ─
print("Running eval dataset through RAG pipeline...")
print("=" * 60)

eval_results=[]

for i ,item in enumerate(eval_questions):
    question=item["question"]
    ground_truth=item["ground_truth"]

    answer=rag_chain.invoke(question)

    retrived_docs=retrival_docs.invoke(question)
    contexts=[doc.page_content for doc in retrived_docs]

    result={
        "question":question,
        "ground_truth":ground_truth,
        "answer":answer,
        "contexts":contexts
    }

    eval_results.append(result)

    print(f"Q{i+1}: {question[:60]}...")
    print(f"  Answer: {answer[:80]}...")
    print(f"  Contexts retrieved: {len(contexts)}")
    print()

# ── SAVE: persist eval results for RAGAS ─────────────────
os.makedirs("rag",exist_ok=True)
with open("rag/eval_results.json","w") as f:
    json.dump(eval_results,f,indent=2)

print("=" * 60)
print(f"Saved {len(eval_results)} eval results to rag/eval_results.json")
print("Run step5_ragas.py next to score these results")