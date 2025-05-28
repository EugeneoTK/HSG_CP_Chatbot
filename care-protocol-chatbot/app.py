import streamlit as st
import json
from openai import OpenAI
import os
st.set_page_config(page_title="Diabetes Protocol Chatbot", layout="wide")
# Load OpenAI API key from secrets
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the Q&A data from the JSON file
@st.cache_data
def load_qa_pairs():
    with open("care-protocol-chatbot/word_protocol_qa.json", "r") as f:
        return json.load(f)

qa_pairs = load_qa_pairs()

# Basic Streamlit setup

st.title("🩺 Diabetes Protocol Chatbot")
st.write("Ask questions based on the care protocol. The bot will ground answers on the extracted Q&A pairs.")

# Function to find the most relevant QA pair
def find_best_match(user_question):
    import difflib
    questions = [item["question"] for item in qa_pairs]
    match = difflib.get_close_matches(user_question, questions, n=1, cutoff=0.4)
    if match:
        for item in qa_pairs:
            if item["question"] == match[0]:
                return item["answer"]
    return None

# Function to rephrase or expand via OpenAI
def ask_openai(question, context):
    prompt = f"""You are a helpful assistant. Answer the question strictly based on the care protocol context.

Context: {context}

Question: {question}
Answer:"""
    try:
        response = client.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error from OpenAI: {str(e)}"

# Chat interface
question = st.text_input("❓ Enter your question:")

if question:
    with st.spinner("Thinking..."):
        matched_answer = find_best_match(question)
        if matched_answer:
            final_answer = ask_openai(question, matched_answer)
            st.markdown("### 💬 Answer")
            st.success(final_answer)
        else:
            st.warning("No relevant match found in the care protocol data.")
