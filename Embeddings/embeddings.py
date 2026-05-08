import openai
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()
client=openai.OpenAI(api_key=os.getenv("OPEN_API_KEY"))

#-------------Step 1 Get first embeddings---------
text = "LangGraph is a framework for building stateful agents"

response=client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)

embedding=response.data[0].embedding
print(f"Text: {text}")
print(f"Embedding dimensions: {len(embedding)}")       # 1536
print(f"First 5 numbers: {embedding[:5]}")
print(f"Type: {type(embedding[0])}")                   # float
print()

# ── PART 2: Cosine similarity from scratch ────────────────
# This is what vector DBs do internally

def cosine_similarity(vec1:list,vec2:list)->float:
    # ── PART 2: Cosine similarity from scratch ────────────────
    # This is what vector DBs do internally — understand it first
    a=np.array(vec1)
    b=np.array(vec2)

    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))


# ── PART 3: Embed multiple texts and compare ──────────────
texts = [
    "LangGraph is a framework for building stateful agents",   # query topic
    "I love eating pizza and pasta",                           # unrelated
    "Agents can reason and plan using LLMs",                   # related
    "CrewAI helps build multi-agent systems",                  # related
    "The weather in Mumbai is very humid",                     # unrelated
]

print("Embedding all texts...")
response2=client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

all_embedding=[item.embedding for item in response2.data]

# ── PART 4: Find most similar to query ───────────────────
query=texts[0]
query_emd=all_embedding[0]

print(f"\nQuery: '{query}'\n")
print("Similarity scores:")
print("-" * 60)

scores=[]

for i,(text,emb) in enumerate(zip(texts[1:],all_embedding[1:]),1):
    score=cosine_similarity(query_emd,emb)
    scores.append((score,text))
    print(f"Score: {score:.4f} | {text}")

# Sort by similarity
scores.sort(reverse=True)
print("\nMost similar to query:")
print(f"  → {scores[0][1]}")
print(f"  Score: {scores[0][0]:.4f}")

# ── PART 5: Semantic search function ─────────────────────
def semantic_search(query:str,documents:list[str],top_k:int=2)->list[tuple[float,str]]:
    """
    Given a query and list of documents,
    return top_k most semantically similar documents.
    This is what ChromaDB and Qdrant do internally.
    """
    # Embed query + all documents in one call
    all_texts=[query]+documents
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=all_texts
    )

    embeddings=[item.embedding for item in response]
    query_emb=embeddings[0]
    doc_embs = embeddings[1:]

    scored=[
        (cosine_similarity(query_emb,doc_embs),doc)
        for doc_emb,doc in zip(doc_embs,documents)
    ]

    return sorted(scored,reverse=True)[:top_k]

documents = [
    "LangGraph uses a state machine to control agent flow",
    "Pizza originated in Naples, Italy in the 18th century",
    "CrewAI agents can collaborate on complex tasks",
    "Mumbai receives heavy rainfall during monsoon season",
    "RAG pipelines retrieve relevant documents before generating answers",
    "The stock market saw gains today due to positive earnings",
    "Qdrant is a vector database optimized for similarity search",
]

query = "How do AI agents work together?"
results = semantic_search(query, documents, top_k=3)

print(f"\n\nSemantic search for: '{query}'")
print("Top 3 results:")
for score, doc in results:
    print(f"  [{score:.4f}] {doc}")

