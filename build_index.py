import os
import datetime
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

DATA_DIR = "data"
INDEX_DIR = "faiss_index"
LOG_FILE = "index_log.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_all_pdfs(data_dir):
    documents = []
    indexed_files = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            path = os.path.join(data_dir, file)
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
            indexed_files.append(file)
    return documents, indexed_files


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def log_indexed_files(indexed_files):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as log:
        log.write(f"[{timestamp}] Indexed files:\n")
        for file in indexed_files:
            log.write(f" - {file}\n")
        log.write("\n")


def rebuild_faiss_index():
    print("Loading documents from", DATA_DIR)
    documents, indexed_files = load_all_pdfs(DATA_DIR)
    print(f"Loaded {len(documents)} pages. Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks. Embedding and indexing...")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"✅ FAISS index rebuilt and saved to '{INDEX_DIR}'")

    log_indexed_files(indexed_files)
    print(f"📜 Indexing log saved to '{LOG_FILE}'")


if __name__ == "__main__":
    rebuild_faiss_index()
