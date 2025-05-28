import streamlit as st
import json
import os
from openai import OpenAI

# This must be the first Streamlit call
st.set_page_config(page_title="Diabetes Protocol Chatbot", layout="wide")

# Load OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load Q&A pairs from JSON file
@st.cache_data
def load_qa_pairs():
    try:
        with open("word_protocol_qa.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

qa_pairs = load_qa_pairs()

# Define OpenAI call
def ask_openai(question, context):
    prompt = f"""You are a protocol-based assistant. Answer ONLY using the content in the care protocol below.
If the answer is not present, say: "The protocol does not specify this information."

Do NOT use external knowledge. Do NOT guess. Be brief and precise.

Protocol:
{context}

Question:
{question}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error from OpenAI: {e}"

# UI
st.title("📘 Diabetes Protocol Chatbot")

question = st.text_input("Ask a question about the Diabetes Care Protocol")

if question:
    # Find best-match context
    context = ""
    for pair in qa_pairs:
        if question.lower() in pair["question"].lower():
            context = pair["answer"]
            break
    if not context and qa_pairs:
        context = qa_pairs[0]["answer"]

    with st.spinner("Consulting the protocol..."):
        response = ask_openai(question, context)

    st.subheader("🧠 Answer")
    st.write(response)
