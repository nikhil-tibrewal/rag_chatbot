import os
from langchain_community.document_loaders import PyPDFLoader  # For reading PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting long texts into smaller chunks
from langchain_community.vectorstores import FAISS  # For storing and retrieving vectorized chunks
from langchain_openai import OpenAIEmbeddings  # Embedding model from OpenAI
from langchain.chains import RetrievalQA  # Retrieval-Augmented Generation chain
from langchain_community.chat_models import ChatOpenAI  # Chat model from OpenAI

# Constants
DATA_DIR = "data"  # Directory where PDF files are stored
INDEX_DIR = "faiss_index"  # Directory to store the FAISS vector index
CHUNK_SIZE = 500  # Size of each text chunk (in tokens)
CHUNK_OVERLAP = 50  # Number of overlapping tokens between chunks


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


def build_vectorstore(chunks):
    """
    Embed all text chunks using OpenAI and save them into a FAISS vector index.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)


def get_vectorstore():
    """
    Load an existing FAISS vector index from disk.
    """
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(INDEX_DIR, embeddings)


def load_qa_chain():
    """
    Create a RetrievalQA chain combining retriever (FAISS) and LLM (OpenAI chat model).
    """
    retriever = get_vectorstore().as_retriever()
    llm = ChatOpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
