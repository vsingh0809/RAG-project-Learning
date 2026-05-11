from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
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
    chunk_size
)