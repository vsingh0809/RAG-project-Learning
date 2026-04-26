import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# STEP 1: YOUR KNOWLEDGE BASE
# (Think of this as "company documents" or "ERP data")
# ─────────────────────────────────────────────
documents = [
    "Our refund policy allows returns within 30 days of purchase with a valid receipt.",
    "Premium subscribers get priority support with response time under 2 hours.",
    "The product is available in three colors: red, blue, and black.",
    "Shipping is free for orders above 999 rupees within India.",
    "To reset your password, go to Settings then Security then Reset Password.",
    "Gold jewelry is hallmarked with BIS certification for quality assurance.",
    "Diamond rings are graded by carat weight, cut, color, and clarity.",
    "All silver items come with a 1-year warranty against tarnishing.",
]

# ─────────────────────────────────────────────
# STEP 2: LOAD EMBEDDING MODEL
# This model converts text → numbers (vectors)
# "all-MiniLM-L6-v2" is small (80MB), fast, and good enough for interviews
# ─────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────
# STEP 3: CREATE VECTOR DATABASE
# ChromaDB stores your documents + their vectors
# It lets you search by MEANING, not just keywords
# ─────────────────────────────────────────────
chroma_client = chromadb.Client()  # in-memory, no setup needed
collection = chroma_client.create_collection(name="company_knowledge")

# Embed all documents and store them
embeddings = embedder.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"Stored {len(documents)} documents in vector DB\n")

# ─────────────────────────────────────────────
# STEP 4: THE RAG FUNCTION
# This is the full pipeline in one function
# ─────────────────────────────────────────────
def rag_query(user_question: str, top_k: int = 2) -> str:
    
    # A. Embed the user's question into a vector
    query_vector = embedder.encode([user_question]).tolist()
    
    # B. Find top_k most similar document chunks
    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k
    )
    
    retrieved_chunks = results["documents"][0]
    distances = results["distances"][0]
    
    # Print what was retrieved (important for debugging in interviews)
    print(f"Question: {user_question}")
    print("Retrieved context:")
    for chunk, score in zip(retrieved_chunks, distances):
        print(f"  [score: {score:.4f}] {chunk}")
    
    # C. Build the augmented prompt
    # This is the "augmentation" step — injecting context into prompt
    context = "\n".join(retrieved_chunks)
    
    prompt = f"""You are a helpful customer support assistant.
Use ONLY the information provided in the context below to answer the question.
If the answer is not present in the context, respond with: "I don't have that information."
Do not make up any information.

Context:
{context}

Question: {user_question}

Answer:"""
    
    # D. Send to LLM for generation
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # low = more factual, less creative
    )
    
    answer = response.choices[0].message.content
    print(f"Answer: {answer}\n")
    return answer


# ─────────────────────────────────────────────
# STEP 5: TEST IT
# ─────────────────────────────────────────────
test_questions = [
    "What is the return policy?",          # Should find answer
    "Tell me about diamond grading",        # Should find answer
    "Do you offer EMI options?",            # NOT in docs — should say so
]

print("=" * 60)
for question in test_questions:
    print("-" * 60)
    rag_query(question)