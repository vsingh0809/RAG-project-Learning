from sentence_transformers import SentenceTransformer
import numpy as np

embedding= SentenceTransformer("")
# ─────────────────────────────────────────────
# PART A: UNDERSTAND WHAT AN EMBEDDING IS
# ─────────────────────────────────────────────
sentences = [
    "I want to return my product",      # similar to refund
    "What is the refund policy?",       # similar to return
    "How do I reset my password?",      # completely different
]

vectors = embedding.encode(sentences)

print("=== EMBEDDING SHAPES ===")
print(f"Each sentence → vector of {vectors.shape[1]} numbers")
print(f"Vector for sentence 1 (first 5 nums): {vectors[0][:5]}\n")

# ─────────────────────────────────────────────
# PART B: COSINE SIMILARITY — THE MATH BEHIND RAG

def cosine_similarity(v1,v2):
     # dot product divided by product of magnitudes
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

sim_12= cosine_similarity(vectors[0],vectors[1])
sim_13=cosine_similarity(vectors[0],vectors[3])

print("=== COSINE SIMILARITY ===")
print(f"'return product' vs 'refund policy': {sim_12:.4f}")   # should be HIGH
print(f"'return product' vs 'reset password': {sim_13:.4f}")  # should be LOW
print("Closer to 1.0 = more similar in meaning\n")

# ─────────────────────────────────────────────
# PART C: FAISS — PRODUCTION VECTOR SEARCH
# ─────────────────────────────────────────────
documents = [
    "Our refund policy allows returns within 30 days.",
    "Diamond rings are graded by carat, cut, color, clarity.",
    "Shipping is free for orders above 999 rupees.",
    "Gold jewelry is hallmarked with BIS certification.",
    "Reset your password in Settings > Security.",
    "Premium subscribers get priority support.",
]

doc_vectors=embedding.encode(documents)