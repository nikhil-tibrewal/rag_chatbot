import streamlit as st
import time
from utils import load_qa_chain, get_vectorstore

# Load environment variables from .env into os.environ.
from dotenv import load_dotenv
load_dotenv()

# Streamlit-based interface

st.title("📚 RAG Chatbot Over Your Documents")

query = st.text_input("Ask your question:")

if query:
    start_time = time.time()
    qa_chain = load_qa_chain()
    result = qa_chain.run(query)
    response_time = round(time.time() - start_time, 2)

    st.markdown("### ✅ Answer")
    st.write(result)
    st.markdown(f"⏱ Response Time: {response_time} seconds")

    with open("logs.txt", "a") as f:
        f.write(f"{query} | {result} | {response_time}\n")
