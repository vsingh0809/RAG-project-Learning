from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
import os

# ──  chunking on a simple text ──────────
sample_text = """
LangGraph is a library for building stateful, multi-actor applications with LLMs.
It extends LangChain with the ability to coordinate multiple chains or agents
across multiple steps of computation in a cyclic manner.

LangGraph is inspired by Pregel and Apache Beam. The public interface
draws inspiration from NetworkX. LangGraph builds on top of LangChain
and uses it internally, though it can be used without LangChain.

Key features of LangGraph include:
- Cycles: LangGraph allows you to define flows that involve cycles
- Persistence: LangGraph comes with built-in persistence
- Human-in-the-loop: Because of persistence, LangGraph agents can be paused
- Streaming support: LangGraph supports streaming of both tokens and states
"""

small_splitter=RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    length_function=len
)

good_splitter=RecursiveCharacterTextSplitter(
     chunk_size=512,
     chunk_overlap=20,
     length_function=len,
     separators=["\n\n","\n", ". ", " ", ""]
)

large_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)

small_chunks=small_splitter.split_text(sample_text)
good_chunks=good_splitter.split_text(sample_text)
large_chunks=large_splitter.split_text(sample_text)

print("=" * 60)
print("CHUNK SIZE COMPARISON")
print("=" * 60)
print(f"Small (150):  {len(small_chunks)} chunks")
print(f"Good  (512):  {len(good_chunks)} chunks")
print(f"Large (1000): {len(large_chunks)} chunks")

print("\n--- Small chunk example (chunk 0) ---")
print(repr(small_chunks[0]))

print("\n--- Good chunk example (chunk 0) ---")
print(repr(good_chunks[0]))

# ── PART 2: Chunking with Document objects + metadata ─────
# In real RAG you always use Document objects, not raw strings
# Document = text content + metadata dict
print("\n" + "=" * 60)
print("CHUNKING WITH METADATA (production pattern)")
print("=" * 60)

documents=[
    Document(
        page_content=sample_text,
        metadata={
            "source":"langchain_docs.txt",
            "page":1,
            "section":"introduction",
            "author":"langchain team"
        }
    ),
    Document(
        page_content="""
        RAG stands for Retrieval Augmented Generation.
        It is a technique that combines information retrieval
        with text generation. Instead of relying purely on
        the LLM's training data, RAG retrieves relevant
        documents and uses them as context for generation.
        This grounds the LLM's answers in real data.
        """,
        metadata={
            "source": "rag_overview.txt",
            "page": 1,
            "section": "definition",
            "author": "AI Research Team"
        }
    )
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    length_function=len,
)

# split_documents preserves metadata in every chunk
chunks = splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")
print()

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:")
    print(f"  Content: {chunk.page_content[:80].strip()}...")
    print(f"  Source:  {chunk.metadata['source']}")
    print(f"  Section: {chunk.metadata['section']}")
    print(f"  Length:  {len(chunk.page_content)} chars")
    print()

# ── PART 3: Load a real PDF and chunk it ──────────────────
# Download any PDF to test — e.g. a research paper
print("=" * 60)
print("LOADING AND CHUNKING A REAL FILE")
print("=" * 60)

# Create a test text file if you don't have a PDF
test_file = "rag/test_document.txt"
os.makedirs("rag", exist_ok=True)

with open(test_file, "w") as f:
    f.write("""
Introduction to Vector Databases

Vector databases are specialized database systems designed to store and
query high-dimensional vectors efficiently. They are the backbone of
modern RAG (Retrieval Augmented Generation) systems.

How Vector Databases Work

When you store a document in a vector database, the text is first
converted to a numerical vector using an embedding model. This vector
captures the semantic meaning of the text. The database stores both
the original text and its vector representation.

When querying, the search term is also converted to a vector.
The database then finds stored vectors that are mathematically
closest to the query vector using metrics like cosine similarity.

Popular Vector Databases

Qdrant is an open-source vector database written in Rust. It offers
high performance and rich filtering capabilities. Qdrant supports
payload (metadata) filtering alongside vector search.

ChromaDB is a simple, open-source vector database popular for
prototyping and small-scale RAG applications. It runs in-memory
or persisted to disk with no external dependencies.

Pinecone is a managed cloud vector database. It handles scaling
automatically but requires a paid account for production use.
    """)


# Load with TextLoader
loader = TextLoader(test_file)
raw_docs = loader.load()
print(f"Loaded {len(raw_docs)} document(s)")
print(f"Total chars: {len(raw_docs[0].page_content)}")

# Chunk it
chunks = splitter.split_documents(raw_docs)
print(f"Produced {len(chunks)} chunks")
print()

print("All chunks with sizes:")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i}: {len(chunk.page_content)} chars | "
          f"'{chunk.page_content[:80].strip()}...'")
    
