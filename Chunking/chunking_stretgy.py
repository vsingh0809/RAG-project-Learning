import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_experimental.text_splitter import SemanticChunker
#Recursive Chunking (Generic RAG)
"""Best for:

blogs
articles
generic text

Uses separator hierarchy:

paragraph → sentence → word"""

text =  """
LangGraph is a framework for stateful AI agents.

It uses nodes and edges to control execution flow.

CrewAI focuses on role-based multi-agent collaboration.

RAG retrieves documents before generation.
"""

text_1 = [
    "Photosynthesis is the process by which green plants and some other organisms use sunlight ",
    "to synthesize foods from carbon dioxide and water. It generally involves the green pigment ",
    "chlorophyll and generates oxygen as a byproduct. ",
    "In 44 BC, Julius Caesar was assassinated by a group of rebellious senators led by Brutus and Cassius. ",
    "This event sparked the final civil wars of the Roman Republic and led to the rise of the Roman Empire. ",
    "To pull a perfect shot of espresso, you need exactly 18 grams of finely ground coffee. The extraction ",
    "time should be between 25 and 30 seconds, yielding a thick, hazelnut-colored crema on top.",
]

# splitter=RecursiveCharacterTextSplitter(
#     chunk_size=80,
#     chunk_overlap=20
# )

# chunks=splitter.split_text(text_1)

# for i, chunk in enumerate(chunks, 1):
#     print(f"\nChunk {i}:")
#     print(chunk)

"""🔥 2️⃣ Semantic Chunking

Best for:

enterprise RAG
mixed-topic docs
high-quality retrieval
🧠 Idea

Split where topic changes."""

sentences = [
    "LangGraph uses graphs for workflows.",
    "Agents can maintain state.",
    "Nodes control execution.",
    "Docker containers package applications.",
    "Docker images are lightweight."
]
text_2 = [
    "Electric vehicles are rapidly replacing traditional combustion engine cars across the world. "
    "The most critical and expensive component of an electric vehicle is its massive lithium-ion battery pack. "
    "These batteries require vast amounts of raw materials, particularly lithium, cobalt, and nickel. "
    "Mining these specific metals often takes place in environmentally sensitive regions, requiring heavy machinery. "
    "Because of this, global supply chains are becoming increasingly strained as countries compete for mining rights. "
    "International trade agreements are currently being rewritten to secure access to these vital rare-earth resources."
]

embedding_object=OllamaEmbeddings(
    model="mxbai-embed-large",
)

embeddings=embedding_object.embed_documents(sentences)

# model = SentenceTransformer("all-MiniLM-L6-v2")

# embeddings = model.encode(sentences)

chunks=[]
current_chunk=[sentences[0]]

THRESHOLD = 0.5

print("Watching the math happen:")
print("-" * 50)

for i in range(1, len(sentences)):
    sim = cosine_similarity(
        [embeddings[i - 1]],
        [embeddings[i]]
    )[0][0]

    # This print statement will show you exactly why it wasn't splitting before!
    print(f"Similarity between sentence {i-1} and {i}: {sim:.4f}")

    if sim < THRESHOLD:
        chunks.append(current_chunk)
        current_chunk = [sentences[i]]
        print("  --> TOPIC CHANGED! CUTTING CHUNK HERE.")
    else:
        current_chunk.append(sentences[i])
        print("  --> Same topic. Grouping together.")

chunks.append(current_chunk)

# ── 5. The Output ────────────────────────────────────────
print("\n" + "=" * 50)
print("FINAL CHUNKS")
print("=" * 50)
for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i}:")
    print(" ".join(chunk))    


#----------------- langchain Semantic_splitter-----------------
sementic_spliiter=SemanticChunker(
    embedding_object,
    breakpoint_threshold_type="percentile"
)

chunks=sementic_spliiter.split_text(text)
 
# ── 5. The Output ────────────────────────────────────────
print("\n" + "=" * 50)
print("FINAL CHUNKS")
print("=" * 50)
for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i}:")
    print(" ".join(chunk)) 