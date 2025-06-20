from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import time
import shutil
import logging
from datetime import datetime, timedelta
import queue
from threading import Lock
import chromadb.api

# Rate limiting configuration
class RateLimiter:
    def __init__(self, calls_per_minute=60):
        self.calls_per_minute = calls_per_minute
        self.calls = queue.Queue()
        self.lock = Lock()
    
    def wait_if_needed(self):
        now = datetime.now()
        with self.lock:
            # Remove calls older than 1 minute
            while not self.calls.empty():
                call_time = self.calls.queue[0]
                if now - call_time > timedelta(minutes=1):
                    self.calls.get()
                else:
                    break
            
            # If at rate limit, wait until oldest call expires
            if self.calls.qsize() >= self.calls_per_minute:
                oldest_call = self.calls.queue[0]
                wait_time = (oldest_call + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    time.sleep(wait_time)
            
            # Add current call
            self.calls.put(now)

# Initialize rate limiter
RATE_LIMITER = RateLimiter(calls_per_minute=50)

load_dotenv()
OPENAI_API = os.getenv("OPENAI_API_KEY")

# Set USER_AGENT for web requests
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "LangChain Fitness Chatbot/1.0")

# Always store Chroma data at project-root (one shared directory)
CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "chroma_db"))
chroma_dir = CHROMA_DIR

# Remove any stale vector-store inside the package directory (left-over from earlier code)
_STALE_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
if os.path.isdir(_STALE_DIR) and os.path.abspath(_STALE_DIR) != CHROMA_DIR:
    try:
        shutil.rmtree(_STALE_DIR)
    except Exception as _e:  # log and ignore – not fatal
        logging.warning(f"[RAG] Could not remove stale Chroma folder {_STALE_DIR}: {_e}")

os.environ.setdefault("ALLOW_RESET", "TRUE")

def get_rag_chain(pdf_dir, urls, max_tokens=1500):
    """
    Create a simple, robust RAG chain that works reliably.
    This function handles all document loading, embedding, and chain creation.
    """
    logging.info(f"[RAG] Initializing RAG chain with PDF_DIR={pdf_dir} and URLs={urls}")
    
    # Initialize OpenAI components
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", api_key=OPENAI_API)
    
    # Set up ChromaDB
    client = chromadb.Client(Settings(persist_directory=chroma_dir, anonymized_telemetry=False))
    
    # Check if collection exists
    collection_exists = False
    try:
        existing_collections = [col.name for col in client.list_collections()]
        collection_exists = "fitness_docs" in existing_collections
        logging.info(f"[RAG] Collection exists: {collection_exists}")
    except Exception as e:
        logging.warning(f"[RAG] Error checking collections: {e}")
        collection_exists = False
    
    # Load and process documents if collection doesn't exist
    if not collection_exists:
        logging.info("[RAG] Building new vectorstore...")
        documents = []
        
        # Load PDFs
        if pdf_dir and os.path.exists(pdf_dir):
            try:
                pdf_loader = DirectoryLoader(pdf_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
                pdf_docs = pdf_loader.load()
                # Remove duplicates
                unique_docs = {}
                for doc in pdf_docs:
                    source = doc.metadata.get('source', 'unknown')
                    if source not in unique_docs:
                        unique_docs[source] = doc
                documents.extend(unique_docs.values())
                logging.info(f"[RAG] Loaded {len(unique_docs)} unique PDFs")
            except Exception as e:
                logging.error(f"[RAG] Error loading PDFs: {e}")
        
        # Load URLs
        if urls:
            try:
                web_loader = WebBaseLoader(urls)
                web_docs = web_loader.load()
                documents.extend(web_docs)
                logging.info(f"[RAG] Loaded {len(web_docs)} web documents")
            except Exception as e:
                logging.error(f"[RAG] Error loading URLs: {e}")
        
        # Split documents
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            split_docs = text_splitter.split_documents(documents)
            logging.info(f"[RAG] Split into {len(split_docs)} chunks")
            
            # Create vectorstore
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                collection_name="fitness_docs",
                persist_directory=chroma_dir
            )
            logging.info(f"[RAG] Created vectorstore with {len(split_docs)} documents")
        else:
            logging.warning("[RAG] No documents found, creating empty vectorstore")
            vectorstore = Chroma(
                collection_name="fitness_docs",
                embedding_function=embeddings,
                persist_directory=chroma_dir
            )
    else:
        # Load existing vectorstore
        logging.info("[RAG] Loading existing vectorstore...")
        vectorstore = Chroma(
            collection_name="fitness_docs",
            embedding_function=embeddings,
            persist_directory=chroma_dir
        )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )

    # Prompt that includes the automatically injected chat history
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert bodybuilding, fitness, and nutrition assistant. "
                    "Use the conversation history to answer meta-questions such as listing or repeating" 
                    "previous user questions. If the current user request is about earlier turns, quote or "
                    "summarise those turns verbatim. Otherwise, answer the question using the retrieved "
                    "fitness context. Cite sources when context is used; otherwise reply with 'General knowledge'."
                ),
            ),
            ("human", "Question: {question}\n\nContext:\n{context}")
        ]
    )

    # Build conversational chain with custom prompt
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
    )
    
    logging.info("[RAG] Conversational RAG chain created successfully with custom prompt")
    return chain

def clear_chromadb(max_retries: int = 5) -> bool:
    """Delete the `fitness_docs` collection (if any) and remove on-disk data.

    The PersistentClient API is used first to drop the collection so that all
    MMAP handles are closed cleanly. After that the directory is removed. On
    Windows we still retry a few times to cope with the OS file-lock latency.
    """
    try:
        # reset global state to avoid instance-conflict errors
        try:
            chromadb.api.global_state.reset_state()
        except Exception:
            pass

        # allow_reset flag needed from Chroma 0.4.22 onwards
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(persist_directory=chroma_dir, allow_reset=True)
        )
        client.reset()  # clear everything atomically (fast path)
        logging.info("[RAG] Chroma client reset – all collections removed")
        return True
    except Exception as e:
        logging.warning(f"[RAG] reset() unavailable – falling back to per-collection delete ({e})")
        try:
            client = chromadb.PersistentClient(path=chroma_dir)
            for col in client.list_collections():
                try:
                    client.delete_collection(name=col.name)  # fully removes collection files
                    logging.info(f"[RAG] Deleted collection {col.name}")
                except Exception as _e:
                    logging.warning(f"[RAG] Could not delete collection {col.name}: {_e}")
            return True
        except Exception as e2:
            logging.error(f"[RAG] Failed fallback Chroma clear: {e2}")
            return False

def cleanup_vector_stores():
    """Clean up all vector store related directories and caches"""
    paths_to_clean = [
        CHROMA_DIR,
        "./.chroma",
        "./.cache",
        "./cache",
        "./web_faiss_index",
        "./combined_faiss_index",
        "./__pycache__",
        os.path.join(os.path.dirname(__file__), "__pycache__")
    ]
    
    for path in paths_to_clean:
        if os.path.exists(path):
            try:
                print(f"Removing {path}")
                shutil.rmtree(path)
            except Exception as e:
                print(f"Error removing {path}: {str(e)}")

def batch_embed(texts, embedder, batch_size: int = 50, max_retries: int = 3):
    """Embed texts in batches with simple retry logic."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        sub_texts = texts[i:i+batch_size]
        for attempt in range(max_retries):
            try:
                RATE_LIMITER.wait_if_needed()
                embeddings.extend(embedder.embed_documents(sub_texts))
                break
            except Exception as e:
                logging.warning(f"[RAG] Embedding batch failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(1)
        else:
            raise RuntimeError("Embedding failed after retries")
    return embeddings
