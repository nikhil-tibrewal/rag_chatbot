import datetime
from utils import load_all_pdfs, split_documents, build_vectorstore

LOG_FILE = "index_log.txt"


def log_indexed_files(indexed_files):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as log:
        log.write(f"[{timestamp}] Indexed files:\n")
        for file in indexed_files:
            log.write(f" - {file}\n")
        log.write("\n")


def rebuild():
    print("Loading documents from data/")
    documents, indexed_files = load_all_pdfs("data")
    print(f"Loaded {len(documents)} pages. Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks. Embedding and indexing...")
    build_vectorstore(chunks)
    print("✅ FAISS index rebuilt and saved.")
    log_indexed_files(indexed_files)
    print(f"📜 Indexing log saved to '{LOG_FILE}'")


if __name__ == "__main__":
    rebuild()
