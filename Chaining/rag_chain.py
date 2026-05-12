from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

# ── Setup: embeddings + LLM ───────────────────────────────
embeddings=OllamaEmbeddings(
    model="mxbai-embed-large"
)

# Use Gemini as LLM — free tier
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

# ── Load existing ChromaDB from Step 3 ────────────────────

vector_store=Chroma(
    embedding_function=embeddings,
    persist_directory="../Vector_Database/chroma_db",
    collection_name="ai_knowledge_base"
)

# ── Manual RAG — understand each step ────────────
# Do this first so you understand what the chain automates
print("=" * 60)
print("PART 1: MANUAL RAG STEP BY STEP")
print("=" * 60)

question = "What is LangGraph and how does it work?"

# Step 1: Retrieve relevant chunks
retrieved_docs=vector_store.similarity_search(question,k=3)
print(f"Retrieved {len(retrieved_docs)} chunks for: '{question}'")
for doc in retrieved_docs:
    print(f"  → [{doc.metadata['source']}] {doc.page_content[:60]}...")
print()

# Step 2: Format context from retrieved chunks
context="\n\n".join([doc.page_content for doc in retrieved_docs])

# Step 3: Build prompt manually
manual_prompt=f"""Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't have that information.

Context:{context}
Question:{question}
Answer:"""

# Step 4: Call LLM
response=llm.invoke(manual_prompt)
print("Manual RAG Answer:")
print(response.content)
print()

# ──--------- RAG Chain using LCEL ─────────────────────────
# Now automate the manual steps using LangChain Expression Language
print("=" * 60)
print("PART 2: RAG CHAIN WITH LCEL (production pattern)")
print("=" * 60)

# Retriever: wraps vectorstore with a clean interface
retriver=vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

#Prompt template for RAG
RAG_PROMPT=ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Answer the question based ONLY on the
provided context. If the answer is not in the context, clearly state
that you don't have that information in your knowledge base.
Always be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""
)

def formate_docs(docs:list[Document])->str:
        """Format retrieved docs into a single context string."""
        return "\n\n".join([
              f"Source:{doc.metadata.get('source','unknown')}\n{doc.page_content}"
              for doc in docs
        ])

# Build the RAG chain using LCEL pipe operator
rag_chain=(
      {
            "context":retriver|formate_docs,  # retrieve + format
            "question":RunnablePassthrough()    # pass question through unchanged  
      }
      |RAG_PROMPT           # build the prompt
      |llm                 
      |StrOutputParser()
                  # extract string from response
)

questions = [
    "What is RAG and why is it useful?",
    "How does ChromaDB work?",
    "What frameworks can I use to build AI agents?",
    "How do I evaluate a RAG pipeline?",
    "What is the capital of Australia?",   # not in knowledge base
]

# for q in questions:
#       print(f"Q:{q}")
#       answer=rag_chain.invoke(q)
#       print(f"A :{answer}")
#       print("-" * 50)


# rag with source citations
print()
print("=" * 60)
print("PART 3: RAG WITH SOURCE CITATIONS")
print("=" * 60)

from langchain_core.runnables import RunnableParallel

# Return both answer AND source documents
rag_chain_with_sources=RunnableParallel(
      {
            "answer":rag_chain,
            "sources":retriver
      }
)

result=rag_chain_with_sources.invoke("what is rag used for")

print(f"Answer: {result['answer']}")
print()
print("Sources used:")
for doc in result['sources']:
    print(f"  - {doc.metadata['source']} (topic: {doc.metadata['topic']})")

# ──: Conversational RAG — multi-turn ───────────────
print()
print("=" * 60)
print("PART 4: CONVERSATIONAL RAG (multi-turn)")
print("=" * 60)    

from langchain_core.messages import HumanMessage, AIMessage

CONV_Rag_Prompt=ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant. Use the context below to answer questions.
Keep track of the conversation history to answer follow-up questions.

Context from knowledge base:
{context}

Conversation history:
{chat_history}

Current question: {question}

Answer:"""
)

conv_rag_chain=(
     {
          "context":retriver|formate_docs,
          "question": RunnablePassthrough(),
          "chat_history": lambda _: chat_history
     }
     |CONV_Rag_Prompt
     |llm
     |StrOutputParser()
)

#Simulate multiturn conversation
chat_history=[]
conversation=[
      "What is LangGraph?",
    "What are its key features?",           # follow-up — tests memory
    "How is it different from CrewAI?",     # comparison question
]

for question in conversation:
     print(f"user:{question}")
     answer=conv_rag_chain.invoke(question)
     print(f"Bot:  {answer}")
     print()
     #store history

     chat_history.extend([
          HumanMessage(content=question),
          AIMessage(content=answer)
     ])



