import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── NON-STREAMING (what you have been doing) ─────────────
print("=== NON-STREAMING ===")
response=client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role":"user","content":"Write a 3 sentence story about a robot."}]
)
print(response.choices[0].message.content)
print()


# ── STREAMING (industry standard) ────────────────────────
print("=== STREAMING ===")

stream=client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role":"user","content":"Write a 3 sentence story about a robot."}],
    stream=True
)


full_response=[]
for chunk in stream:
    token=chunk.choices[0].delta.content

    if token is not None:
        print(token, end="")
        full_response+=token


print()
print(f"\n Total chars recieved:{len(full_response)}")

# ── INSPECT A RAW CHUNK ───────────────────────────────────
print("\n=== WHAT A CHUNK LOOKS LIKE ===")

stream2=client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Say: hello world"}],
    stream=True
)

for i, chunk in enumerate(stream2):
    print(f"Chunk {i}: {repr(chunk)}") 
    if i>4 :
      print(".........(more chunk follow)")
      for _ in stream2 : pass
      break

