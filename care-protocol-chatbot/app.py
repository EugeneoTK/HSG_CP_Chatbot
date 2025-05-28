import streamlit as st
import json
import openai
from sentence_transformers import SentenceTransformer, util

# Load your protocol Q&A
with open("protocol_qa.json", "r", encoding="utf-8") as f:
    qa_list = json.load(f)

# Embed questions using a local embedding model
@st.cache_resource
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    questions = [q["question"] for q in qa_list]
    embeddings = model.encode(questions, convert_to_tensor=True)
    return model, embeddings, questions

model, embeddings, questions = load_embeddings()

# UI
st.title("🩺 Primary Care Protocol Chatbot")
st.write("Ask a question and I’ll respond based only on your uploaded care protocol.")

user_input = st.text_input("Your question")

if user_input:
    # Embed the user query
    query_embedding = model.encode(user_input, convert_to_tensor=True)

    # Semantic search
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_idx = scores.argmax().item()

    best_match = qa_list[top_idx]
    st.markdown(f"**Q:** {best_match['question']}")
    st.markdown(f"**A:** {best_match['answer']}")
