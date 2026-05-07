from dotenv import load_dotenv
from groq import Groq
import os
import streamlit as st


load_dotenv()
client=Groq(api_key=os.getenv("GROQ_API_KEY"))

#-----------page config-------------
st.set_page_config(
    page_title="Ai Chat",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Streaming AI Chat")
st.caption("Powered by Grok")

#---------------session state--------
# Streamlit reruns the entire script on every interaction.
# session_state is the ONLY way to persist data across reruns.
if"messages" not in st.session_state:
    st.session_state.messages=[]

#-----------display existing messages-----------
def stream_response(messages:list[dict]):
     """
    Generator compatible with st.write_stream().
    Yields string tokens — Streamlit renders them as they arrive.
    Same stream_chat() pattern from Step 2, adapted for Streamlit.
    """
     stream=client.chat.completions.create(
          model="llama-3.3-70b-versatile",
          messages=messages,
          stream=True
     )

     for chunk in stream:
          token=chunk.choices[0].delta.content
          if token is not None:
               yield token

if prompt := st.chat_input("ask anything"):
     
     # 1. Add user message to history + display immediately
     st.session_state.messages.append({"role":"user","content":prompt})
     with st.chat_message("user"):
          st.markdown(prompt)
    
     api_messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        *st.session_state.messages    # full conversation history
    ]
     
     # 3. Stream the assistant response into a chat bubble
     with st.chat_message("assistant"):
          # st.write_stream() handles the generator
        # renders tokens as they arrive AND returns complete string
          full_response=st.write_stream(stream_response(api_messages))
     
     # 4. Save completed response to history for next turn
     st.session_state.messages.append({
          "role":"assistant",
          "content":full_response
     })

# ── SIDEBAR: stats + controls ─────────────────────────────
with st.sidebar:
     st.header("session info")
     total=len(st.session_state.messages)
     turns=len([m for m in st.session_state.messages if m["role"] == "user"])
     st.metric("Total messages", total)
     st.metric("Conversation turns", turns)

     st.divider()

     if st.button("🗑️ Clear conversation"):
          st.session_state.messages=[]
          st.rerun()

     st.divider()
     st.caption("Built with OpenAI streaming + Streamlit")     


