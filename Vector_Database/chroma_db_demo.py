import os
import shutil
from dotenv import load_dotenv
import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

load_dotenv()


embeddings= OllamaEmbeddings(
    model="mxbai-embed-large",
)

#------creating chromaDB for documents-----------
documents=[
    Document(
        page_content="LangGraph is a library for building stateful multi-actor applications. It uses a state machine where nodes are functions and edges control flow between them.",
        metadata={"source": "langchain_docs", "topic": "agents", "page": 1}
    ),
    Document(
        page_content="RAG stands for Retrieval Augmented Generation. It retrieves relevant documents from a vector store and uses them as context for the LLM to generate grounded answers.",
        metadata={"source": "rag_guide", "topic": "rag", "page": 1}
    ),
    Document(
        page_content="ChromaDB is an open-source vector database. It stores embeddings and allows fast similarity search. It can run in-memory or persist to disk.",
        metadata={"source": "chromadb_docs", "topic": "vectordb", "page": 1}
    ),
    Document(
        page_content="CrewAI is a framework for orchestrating role-playing autonomous AI agents. Agents have roles, goals, and backstories. They work together to complete complex tasks.",
        metadata={"source": "crewai_docs", "topic": "agents", "page": 1}
    ),
    Document(
        page_content="Qdrant is a vector database written in Rust. It offers high performance similarity search with rich filtering on payload metadata. Good for production RAG systems.",
        metadata={"source": "qdrant_docs", "topic": "vectordb", "page": 1}
    ),
    Document(
        page_content="RAGAS is a framework for evaluating RAG pipelines. It measures faithfulness, answer relevancy, context precision, and context recall using LLM-as-judge.",
        metadata={"source": "ragas_docs", "topic": "evaluation", "page": 1}
    ),
    Document(
        page_content="Prompt engineering involves designing inputs to LLMs to get better outputs. Techniques include chain-of-thought, few-shot examples, and ReAct patterns.",
        metadata={"source": "prompt_guide", "topic": "prompting", "page": 1}
    ),
    Document(
        page_content="Function calling allows LLMs to request execution of predefined functions. The LLM returns structured JSON with function name and arguments. Your code runs the function.",
        metadata={"source": "openai_docs", "topic": "function_calling", "page": 1}
    ),
]

# ── Store in ChromaDB with persistence ────────────────────
PERSIST_DIR="./chroma_db"

# Delete old data before creating fresh
# if os.path.exists(PERSIST_DIR):
#     shutil.rmtree(PERSIST_DIR)
#     print("Cleared old ChromaDB data")

# from_documents: chunks + embeds + stores in one call
vector_store=Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    #persist_directory=PERSIST_DIR,
    collection_name="ai_knowledge_base"
)

print(f"Stored {len(documents)} documents in ChromaDB")
print(f"Persisted to: {PERSIST_DIR}")
print()

# ──------- Basic similarity search ──────────────────────
print("=" * 60)
print("BASIC SIMILARITY SEARCH")
print("=" * 60)

query="How doAI agents collaborate and work together?"
results=vector_store.similarity_search(query,k=3)

print(f"Query: '{query}'")
print(f"Top {len(results)} results:\n")

for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"  Content: {doc.page_content[:100]}...")
    print(f"  Source:  {doc.metadata['source']}")
    print(f"  Topic:   {doc.metadata['topic']}")
    print()

# ── PART 3: Search with scores ────────────────────────────
print("=" * 60)
print("SEARCH WITH SIMILARITY SCORES")
print("=" * 60)

results_with_scores=vector_store.similarity_search_with_score(query,k=4)

for doc, score in results_with_scores:
    # Chroma returns L2 distance — lower = more similar
    print(f"Score: {score:.4f} | {doc.page_content[:80]}...")
print()

# ── PART 4: Metadata filtering ────────────────────────────
# This is powerful — search only within a category
print("=" * 60)
print("METADATA FILTERING")
print("=" * 60)

# Only search documents where topic == "agents"

agent_results=vector_store.similarity_search(
    query="How do agents works",
    k=3,
    filter={"topic":"agents"}
)

print("Filtered to topic='agents' only:")
for doc in agent_results:
    print(f"  [{doc.metadata['topic']}] {doc.page_content[:80]}...")
print()

# Only search vectordb docs

db_results=vector_store.similarity_search(
    query="which database to use?",
    k=3,
    filter={"topic":"vectordb"}
)

print("Filtered to topic='vectordb' only:")
for doc in db_results:
    print(f"  [{doc.metadata['source']}] {doc.page_content[:80]}...")
print()

# ── PART 5: Load existing ChromaDB (no re-embedding) ─────
print("=" * 60)
print("LOADING EXISTING CHROMADB (no re-embedding)")
print("=" * 60)

loaded_store=Chroma(
   # persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name="ai_knowledge_base"
)

results=loaded_store.similarity_search("what is rag",k=2)

print("Loaded existing store. Search results:")
for doc in results:
    print(f"  {doc.page_content[:100]}...")

# ── PART 6: Add new documents to existing store ───────────
print()
print("=" * 60)
print("ADDING NEW DOCUMENTS TO EXISTING STORE")
print("=" * 60)

new_doc=Document(
    page_content="FastAPI is a modern Python web framework for building APIs. It uses async/await and Pydantic for validation. Perfect for wrapping AI systems.",
    metadata={"source": "fastapi_docs", "topic": "deployment", "page": 1}
)

loaded_store.add_documents([new_doc])
print("Added 1 new document")

# Verify it was added
results = loaded_store.similarity_search("How to deploy AI APIs?", k=2)
print("Search after adding FastAPI doc:")
for doc in results:
    print(f"  [{doc.metadata['source']}] {doc.page_content[:80]}...")

results = loaded_store.similarity_search_with_score("How to deploy AI APIs?", k=2)
for doc, score in results:
    # Chroma returns L2 distance — lower = more similar
    print(f"Score: {score:.4f} | {doc.page_content[:80]}...")
    print()
