import json
import os
from dataclasses import dataclass
from groq import Groq
from dotenv import load_dotenv
import time

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── StreamResult: capture metadata alongside content ──────
# In production you always want timing + token count + cos

@dataclass
class StreamResult:
    content:str=""
    token_count:int=0
    duration_ms :float=0.0
    model:str=""

# ── CORE PATTERN: reusable streaming generator ────────────
# This is the function you will copy into every project.
# It only yields tokens — caller decides what to do with them.

def stream_chat(messages:list[dict],model:str="llama-3.3-70b-versatile",system:str|None=None):
     """
    Reusable streaming generator.
    Usage:
        for token in stream_chat(messages):
            print(token, end="", flush=True)
    """
     if messages:
          messages=[{"role":"system","content":system}]+messages

     stream= client.chat.completions.create(
         model=model,
         messages=messages,
         stream=True
              )    
    
     for chunk in stream:
          token=chunk.choices[0].delta.content

          if token is not None:
               yield token          

# ── stream_with_stats: stream + capture metadata ──────────
def stream_with_stats(messages: list[dict], model: str = "llama-3.3-70b-versatile",system: str | None = None,)->StreamResult:
     """
    Streams to terminal AND returns StreamResult with metadata.
    Use when you need real-time display + the full response object.
    """
     result=StreamResult(model=model)
     start=time.perf_counter()

     print("\nAssistant: ", end="", flush=True)

     for token in stream_chat(messages,model=model,system=system):
        print(token, end="", flush=True)
        result.content+=token
        result.token_count+=1  # approximate: 1 chunk ≈ 1 token

     result.duration_ms=(time.perf_counter()-start)*1000    
     print()
    # Cost: gpt-4o-mini output ≈ $0.60 per 1M tokens
     cost = result.token_count * 0.00000060
     print(f"  [{result.token_count} tokens · {result.duration_ms:.0f}ms · ~${cost:.6f}]")

     return result

# ── stream_typewriter: deliberate delay for visual effect ─
def stream_typewriter(messages: list[dict],delay: float = 0.03)->str:
         """Streaming with typewriter delay. Good for demos."""
         full=""
         print("\nAssistence:",end="",flush=True)
         for token in stream_chat(messages,system="you are a helpfull AI assitent"):
               print(token, end="", flush=True)
               full+=token
               time.sleep(delay)
         print()
         return full

# ── stream_with_spinner: shows spinner until first token ──
def stream_with_spinner(message:list[dict])->str:
     """
    Professional UX pattern:
    Show spinner while LLM is thinking (before first token),
    then switch to streaming once tokens start arriving.
    """
     import threading, itertools
     spinner_active=True
     spinner_chars = itertools.cycle(['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'])

     def spin():
          while spinner_active:
            print(f"\r  Thinking {next(spinner_chars)} ", end="", flush=True)
            time.sleep(0.1)
     thread = threading.Thread(target=spin, daemon=True)
     thread.start()

     full=""
     first=True
     for token in stream_chat(message,system="You are a helpful AI assistant. Be concise"):
          if first:
               spinner_active=False
               time.sleep(0.15)
               print(f"\rAssistant: ", end="", flush=True)
               first=False
          print(token,end="",flush=True)  
          full+=token
     print()
     return full     

 # ── RUN ALL 3 PATTERNS ────────────────────────────────────
msgs = [{"role": "user", "content": "Explain what a RAG pipeline is in 2 sentences."}]

# print("=" * 60)
# print("PATTERN 1: stream_with_stats()")
# print("=" * 60)
# result = stream_with_stats(msgs, system="You are a helpful AI assistant. Be concise.")
# print(f"Captured full response: {len(result.content)} chars, {result.token_count} tokens")

print("\n" + "=" * 60)
print("PATTERN 2: stream_typewriter()")
print("=" * 60)
#stream_typewriter(msgs, delay=0.03)

print("\n" + "=" * 60)
print("PATTERN 3: stream_with_spinner()")
print("=" * 60)
stream_with_spinner(msgs)    

     
          
     

