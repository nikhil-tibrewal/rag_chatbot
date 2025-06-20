import os
from langchain_community.document_loaders import PyPDFLoader  # For reading PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting long texts into smaller chunks
from langchain_community.vectorstores import FAISS  # For storing and retrieving vectorized chunks
from langchain_openai import OpenAIEmbeddings  # Embedding model from OpenAI
from langchain.chains import RetrievalQA  # Retrieval-Augmented Generation chain
from langchain_community.chat_models import ChatOpenAI  # Chat model from OpenAI
from tqdm import tqdm
from more_itertools import chunked  # pip install more-itertools if not already installed

# Constants
DATA_DIR = "data"  # Directory where PDF files are stored
INDEX_DIR = "faiss_index"  # Directory to store the FAISS vector index
CHUNK_SIZE = 500  # Size of each text chunk (in tokens)
CHUNK_OVERLAP = 50  # Number of overlapping tokens between chunks

# Load environment variables from .env into os.environ.
from dotenv import load_dotenv
load_dotenv()

def load_all_pdfs(data_dir=DATA_DIR):
    """
    Load all PDF files from the specified directory and return their combined documents.
    Also return the list of indexed filenames for logging purposes.
    """
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
    """
    Split documents into manageable chunks with overlap for context preservation.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks, batch_size=100):
    """
    Embed all text chunks using OpenAI and save them into a FAISS vector index.

    Parameters:
    - chunks (List[Document]): List of LangChain Document objects.
    - batch_size (int): Number of chunks to embed per API call to speed up the process.
    """
    print(f"\nTotal chunks to embed: {len(chunks)}")

    # Initialize OpenAI embedding model
    embeddings = OpenAIEmbeddings()

    # Extract texts from the Document objects
    texts = [doc.page_content for doc in chunks]

    # Prepare to collect all embeddings
    all_embeddings = []
    print(f"Embedding {len(texts)} chunks in batches of {batch_size}...")

    # Run batch embedding with progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    print("Embeddings complete. Building FAISS index...")

    # Reconstruct Document objects with the embeddings and metadata
    metadatas = [doc.metadata for doc in chunks]
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    # Save index locally
    vectorstore.save_local(INDEX_DIR)
    print(f"FAISS index saved to: {INDEX_DIR}\n")


def get_vectorstore():
    """
    Load an existing FAISS vector index from disk.

    Note: FAISS uses Pickle internally. To enable loading the index,
    we must explicitly allow deserialization by setting `allow_dangerous_deserialization=True`.
    Only do this if you trust the source of the index file (i.e., you built it yourself).
    """
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


def load_qa_chain():
    """
    Create a RetrievalQA chain combining retriever (FAISS) and LLM (OpenAI chat model).
    """
    retriever = get_vectorstore().as_retriever()
    llm = ChatOpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
