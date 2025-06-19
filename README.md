# 🧠 RAG Chatbot with LangChain + OpenAI

This project is a Retrieval-Augmented Generation (RAG) chatbot built using LangChain, OpenAI, and FAISS, with a Streamlit UI and basic logging for observability.

## 🏗 Stack
- **LangChain** for chaining LLM + retriever
- **OpenAI** for LLM API
- **FAISS** as vector store
- **Streamlit** for UI
- **PDF** as knowledge base
- **LangSmith or logging** for observability

## 🚀 Features
- Ingests your PDF documents
- Allows natural language Q&A
- Logs query, response, and response time

## 📦 Installation
```bash
pip install -r requirements.txt
```

## ▶️ Running the App
```bash
streamlit run app.py
```

## 🗂 Folder Structure
```
rag_chatbot/
├── app.py
├── utils.py
├── requirements.txt
├── README.md
├── data/
│   └── your_file_1.pdf
└── faiss_index/
```

## 📈 Observability
Basic logging is enabled to `logs.txt`. You can add LangSmith support later for advanced tracing.

## 📚 Powered by
- [LangChain](https://www.langchain.com/)
- [OpenAI](https://platform.openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)

## GCP Instance Config

- Settings:
	- Name: rag-chatbot
	- Machine Type: e2-micro (2 vCPU, 1 GB RAM)
	- Boot Disk: Ubuntu 22.04 LTS (20 GB)
	- Allow HTTP traffic
