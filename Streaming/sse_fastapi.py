import json
import os
import uvicorn
from dataclasses import dataclass
from groq import Groq, APIError
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
 

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Streaming LLM API")

# Allow all origins for local dev — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],   
)

class ChatMessage(BaseModel):
    role:str
    content:str

class ChatRequest(BaseModel):
    message:list[ChatMessage]
    model:str="llama-3.3-70b-versatile"
    system:str="you are a helpfull assistent."

# ── SSE GENERATOR ─────────────────────────────────────────
# This is the exact pattern used by OpenAI, Anthropic, and
# every production AI API. Learn this pattern cold.
def sse_generator(request:ChatRequest):
     """
    Async generator that yields SSE-formatted chunks.
    SSE format: 'data: {json}\n\n'
    The double newline is required by the SSE spec.
    """
     message=[{"role":"system","content":request.system}]  
     message+=[m.model_dump() for m in request.message] 


     try:
         stream=client.chat.completions.create(
             model=request.model,
             messages=message,
             stream=True
         )

         for chunk in stream:
             token = chunk.choices[0].delta.content
             if token is not None:
                 payload=json.dumps({"token":token,"done":False})
                 yield f"data: {payload}\n\n"

         # Final event — client knows stream has ended
         yield f"data:{json.dumps({'token':'','done':True})}\n\n"          

     except APIError as e:
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"


# ── STREAMING ENDPOINT ────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(request:ChatRequest):
     """
    Stream LLM response as Server-Sent Events.
"""
     return StreamingResponse(
         sse_generator(request),
         media_type="application/json",
         headers={
             "Cache-Control":"no-cache",
             "X-Accel-Buffering": "no",   # disables nginx buffering in prod
            "Connection": "keep-alive",
         }
         )

# ── NON-STREAMING ENDPOINT (for comparison) ───────────────
@app.post("/chat/complete")
async def chat_complete(request: ChatRequest):
    messages = [{"role": "system", "content": request.system}]
    messages += [m.model_dump() for m in request.message]
    response = client.chat.completions.create(
        model=request.model,
        messages=messages
    )
    return {"content": response.choices[0].message.content}

if __name__ == "__main__":
    uvicorn.run(
        "sse_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )