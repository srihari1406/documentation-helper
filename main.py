import streamlit as st
from typing import Any, Dict, List
from backend.core import run_llm
import os

def _format_sources(context_doc: List[Any]):
    return [str((meta.get("source", "unknown"))) for doc in (context_doc or []) if (meta:= (getattr(doc, "metadata", None) or {})) is not None]

st.set_page_config(page_title="Langchain Documentation Helper", page_icon="📚", layout="centered")
st.title("Langchain Documentation Helper")

with st.sidebar:
    st.subheader("session")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", 
         "content": "Hello! I'm here to help you with the Langchain documentation. Ask me anything about Langchain, and I'll do my best to provide you with accurate and helpful information based on the documentation.",
         "sources": ["www.langchain.com"]}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")
prompt = st.chat_input("Ask a question about the Langchain...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer","")).strip() or "Sorry, I couldn't find an answer to your question in the documentation."
                sources = _format_sources(result.get("context", []))
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

        except Exception as e:
            st.markdown("Sorry, something went wrong while processing your request.")
            st.error(str(e))