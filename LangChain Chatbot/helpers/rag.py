from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from data.extraction import extract_text_from_pdfs
from helpers.baseretriever import EnhancedRetriever
from helpers.monitoring import validate_embedding_dimensions, validate_embeddings, log_embedding_operation
import os
from dotenv import load_dotenv
import tiktoken
from typing import List, Dict, Any, Union
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction
from langchain_openai import ChatOpenAI
import time
import shutil
import numpy as np
import logging
from threading import Lock
from datetime import datetime, timedelta
import queue
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate

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
RATE_LIMITER = RateLimiter(calls_per_minute=50)  # Conservative limit

load_dotenv()
OPENAI_API = os.getenv("OPENAI_API_KEY")

# Set USER_AGENT for web requests
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "LangChain Fitness Chatbot/1.0")
os.environ["REQUESTS_CA_BUNDLE"] = os.getenv("REQUESTS_CA_BUNDLE", "")  # Set empty if not provided

def num_tokens_from_string(string: str, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))

def batch_embed(texts, embedder, batch_size=100, max_retries=3):
    """
    Embed texts in batches with retry logic and error handling.
    
    Args:
        texts: List of texts to embed
        embedder: Embedding model instance
        batch_size: Maximum batch size (default: 100)
        max_retries: Maximum number of retries for failed batches (default: 3)
    
    Returns:
        List of embeddings
    """
    from helpers.monitoring import validate_embeddings, log_embedding_operation
    
    if not texts:
        logging.warning("No texts provided for embedding")
        return []
    
    # Ensure batch size doesn't exceed OpenAI's limit
    MAX_BATCH_SIZE = 166  # OpenAI's maximum batch size
    batch_size = min(batch_size, MAX_BATCH_SIZE)
    
    results = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    retry_delay = 1  # Initial retry delay in seconds
    
    try:
        # Get embedding dimension from a sample
        sample_text = str(texts[0])  # Ensure string type
        RATE_LIMITER.wait_if_needed()  # Rate limit check
        sample_embedding = embedder.embed_query(sample_text)
        expected_dim = len(sample_embedding)
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            current_batch = (i // batch_size) + 1
            
            # Ensure batch doesn't exceed maximum size
            if len(batch) > MAX_BATCH_SIZE:
                logging.warning(f"Batch size {len(batch)} exceeds maximum {MAX_BATCH_SIZE}, truncating")
                batch = batch[:MAX_BATCH_SIZE]
            
            # Ensure all texts are strings
            batch = [str(text) for text in batch]
            
            for attempt in range(max_retries):
                try:
                    logging.info(f"Processing batch {current_batch}/{total_batches} (size: {len(batch)})")
                    RATE_LIMITER.wait_if_needed()  # Rate limit check
                    batch_results = embedder.embed_documents(batch)
                    
                    # Validate batch results
                    validate_embeddings(batch_results, expected_dim)
                    
                    results.extend(batch_results)
                    log_embedding_operation(
                        "batch_embed",
                        len(batch),
                        len(texts),
                        success=True
                    )
                    logging.info(f"Successfully embedded batch {current_batch}")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        log_embedding_operation(
                            "batch_embed",
                            len(batch),
                            len(texts),
                            success=False,
                            error=str(e)
                        )
                        logging.error(f"Failed to embed batch {current_batch} after {max_retries} attempts: {str(e)}")
                        raise RuntimeError(f"Failed to embed batch after {max_retries} attempts: {str(e)}")
                    logging.warning(f"Attempt {attempt + 1} failed for batch {current_batch}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        
        # Validate final results
        validate_embeddings(results, expected_dim)
        return results
        
    except Exception as e:
        log_embedding_operation(
            "batch_embed",
            batch_size,
            len(texts),
            success=False,
            error=str(e)
        )
        raise RuntimeError(f"Failed to process embeddings: {str(e)}")

class OpenAIEmbeddingFunction(EmbeddingFunction):
    """Wrapper for OpenAI embeddings to match ChromaDB's interface."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API)
        self.batch_size = 50  # Conservative batch size for ChromaDB interface
    
    def __call__(self, input: Documents) -> List[List[float]]:
        """Process documents in batches with proper error handling."""
        if not input:
            return []
            
        if isinstance(input, str):
            input = [input]
            
        try:
            # Use the improved batch_embed function with conservative batch size
            return batch_embed(input, self.embeddings, batch_size=self.batch_size)
        except Exception as e:
            logging.error(f"Error in OpenAIEmbeddingFunction: {str(e)}")
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

class DummyEmbeddingFunction(EmbeddingFunction):
    """A dummy embedding function for Chroma that only handles queries."""
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API)
    def __call__(self, input: Documents) -> List[List[float]]:
        if isinstance(input, str):
            input = [input]
        return [self.embeddings.embed_query(text) for text in input]

def get_combined_retriever(pdf_dir: str, urls: List[str], max_tokens: int = 500):
    logging.info(f"[RAG] Initializing knowledge base with PDF_DIR={pdf_dir} and URLs={urls}")
    # Initialize embeddings
    embedder = OpenAIEmbeddings(api_key=OPENAI_API)
    query_embedder = QueryEmbeddingFunction(OPENAI_API)
    chroma_dir = "./chroma_db"
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
    client = chromadb.Client(Settings(persist_directory=chroma_dir, anonymized_telemetry=False))
    
    # Load and process documents
    documents = []
    loaded_pdfs = set()  # Use a set to track unique PDFs
    if pdf_dir and os.path.exists(pdf_dir):
        pdf_loader = DirectoryLoader(pdf_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        pdf_docs = pdf_loader.load()
        
        # Deduplicate PDFs based on source
        unique_docs = {}
        for doc in pdf_docs:
            source = doc.metadata.get('source', 'unknown')
            if source not in unique_docs:
                unique_docs[source] = doc
                loaded_pdfs.add(source)
        
        documents.extend(unique_docs.values())
        logging.info(f"[RAG] Loaded unique PDFs: {list(loaded_pdfs)}")
    else:
        logging.warning(f"[RAG] PDF directory not found: {pdf_dir}")
    
    loaded_urls = []
    if urls:
        web_loader = WebBaseLoader(urls)
        web_docs = web_loader.load()
        documents.extend(web_docs)
        loaded_urls = [doc.metadata.get('source', 'unknown') for doc in web_docs]
        logging.info(f"[RAG] Loaded URLs: {loaded_urls}")
    else:
        logging.warning("[RAG] No URLs provided.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(documents) if documents else []
    if split_docs:
        try:
            # Validate embedding dimensions with a sample
            sample_text = split_docs[0].page_content
            sample_embedding = embedder.embed_query(sample_text)
            embedding_dim = len(sample_embedding)
            validate_embedding_dimensions(embedding_dim, embedding_dim)
            
            texts = [doc.page_content for doc in split_docs]
            metadatas = [doc.metadata for doc in split_docs]
            ids = [f"doc_{i}" for i in range(len(split_docs))]
            
            # Use the improved batch_embed function
            embeddings = batch_embed(texts, embedder, batch_size=100)
            
            collection = client.create_collection(
                name="fitness_docs",
                metadata={"hnsw:space": "cosine"},
                embedding_function=None
            )
            
            # Add documents in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                    embeddings=batch_embeddings
                )
            
            vectorstore = Chroma(
                client=client,
                collection_name="fitness_docs",
                embedding_function=query_embedder,
                persist_directory=chroma_dir
            )
            logging.info(f"[RAG] Successfully initialized vectorstore with {len(split_docs)} documents.")
        except Exception as e:
            logging.error(f"[RAG] Error during document processing: {str(e)}")
            raise
    else:
        logging.error("[RAG] No documents loaded for initial knowledge base! Check PDF_DIR and URLs.")
        collection = client.create_collection(
            name="fitness_docs",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None
        )
        vectorstore = Chroma(
            client=client,
            collection_name="fitness_docs",
            embedding_function=query_embedder,
            persist_directory=chroma_dir
        )
    
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    compressor = LLMChainExtractor.from_llm(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    return EnhancedRetriever(
        base_retriever=compression_retriever,
        max_tokens=max_tokens
    )

def cleanup_vector_stores():
    """Clean up all vector store related directories and caches"""
    paths_to_clean = [
        "./chroma_db",
        "./.chroma",
        "./.cache",
        "./cache",
        "./__pycache__",
        "./helpers/__pycache__"
    ]
    
    for path in paths_to_clean:
        if os.path.exists(path):
            try:
                print(f"Removing {path}")
                shutil.rmtree(path)
            except Exception as e:
                print(f"Error removing {path}: {str(e)}")

class QueryEmbeddingFunction:
    """Wrapper for OpenAI embeddings to handle query embedding."""
    def __init__(self, api_key):
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
    
    def __call__(self, input: Union[str, List[str], Documents]) -> List[List[float]]:
        """Handle both string and document inputs for embedding."""
        if isinstance(input, (str, bytes)):
            RATE_LIMITER.wait_if_needed()
            return [self.embeddings.embed_query(str(input))]
        elif isinstance(input, list):
            results = []
            for text in input:
                RATE_LIMITER.wait_if_needed()
                results.append(self.embeddings.embed_query(str(text)))
            return results
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
    
    def embed_query(self, text: Union[str, bytes]) -> List[float]:
        """Embed a single query text."""
        RATE_LIMITER.wait_if_needed()
        return self.embeddings.embed_query(str(text))

def create_embeddings_from_texts(texts, embeddings, save_path=None):
    """Create embeddings from texts using ChromaDB"""
    documents = []
    for txt in texts:
        documents.append({"page_content": txt['content'], "metadata": {"source": txt['file_name']}})

    # Create ChromaDB collection
    client = chromadb.Client(Settings(
        persist_directory="./chroma_db",
        anonymized_telemetry=False
    ))
    
    collection = client.get_or_create_collection(
        name="fitness_knowledge",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add documents to collection
    collection.add(
        documents=[doc['page_content'] for doc in documents],
        metadatas=[doc['metadata'] for doc in documents],
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    
    # Create and return vectorstore
    vectorstore = Chroma(
        client=client,
        collection_name="fitness_knowledge",
        embedding_function=embeddings
    )
    
    return vectorstore

def get_rag_chain(pdf_dir, urls, max_tokens=1500):
    logging.info(f"[RAG] Initializing minimal RAG chain with PDF_DIR={pdf_dir} and URLs={urls}")
    embedder = OpenAIEmbeddings(api_key=OPENAI_API)
    chroma_dir = "./chroma_db"
    client = chromadb.Client(Settings(persist_directory=chroma_dir, anonymized_telemetry=False))

    # Only build and embed if the vectorstore does not exist
    if not os.path.exists(chroma_dir) or not os.listdir(chroma_dir):
        documents = []
        loaded_pdfs = set()
        if pdf_dir and os.path.exists(pdf_dir):
            pdf_loader = DirectoryLoader(pdf_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
            pdf_docs = pdf_loader.load()
            unique_docs = {}
            for doc in pdf_docs:
                source = doc.metadata.get('source', 'unknown')
                if source not in unique_docs:
                    unique_docs[source] = doc
                    loaded_pdfs.add(source)
            documents.extend(unique_docs.values())
        if urls:
            web_loader = WebBaseLoader(urls)
            web_docs = web_loader.load()
            documents.extend(web_docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents) if documents else []
        texts = [doc.page_content for doc in split_docs]
        metadatas = [doc.metadata for doc in split_docs]
        ids = [f"doc_{i}" for i in range(len(split_docs))]

        if split_docs:
            collection = client.create_collection(
                name="fitness_docs",
                metadata={"hnsw:space": "cosine"},
                embedding_function=None
            )
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                batch_embeddings = embedder.embed_documents(batch_texts)
                collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                    embeddings=batch_embeddings
                )
    # Always load the vectorstore
    vectorstore = Chroma(
        client=client,
        collection_name="fitness_docs",
        embedding_function=embedder,
        persist_directory=chroma_dir
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", api_key=OPENAI_API)

    # Detailed system prompt
    system_prompt = (
        "You are an expert bodybuilding, fitness, and nutrition assistant.\n"
        "Your job is to answer user questions using the following sources, in order of priority:\n"
        "1. User-Uploaded Documents and Links: If the user has uploaded any documents (PDFs, text files) or provided custom links, always check these first for relevant information. If you find an answer here, cite the specific document or link as the source.\n"
        "2. Main Knowledge Base: If the answer is not found in the user-uploaded materials, search the main knowledge base, which includes trusted fitness and nutrition resources (e.g., scientific articles, reputable websites, and expert guides). If you find an answer here, cite the specific PDF or URL as the source.\n"
        "3. General Knowledge: If neither the user-uploaded materials nor the main knowledge base contain the answer, use your own knowledge as a fitness and nutrition expert to provide a helpful, safe, and evidence-based answer. Clearly state when you are answering from general knowledge and not from a specific source.\n"
        "If the user asks you to repeat, summarize, or list their previous questions, use the list below and quote the previous user questions word-for-word, in order. If the user asks for a question in quotation marks, wrap the exact question in double quotes.\n"
        "Additional Instructions: Always answer in a friendly, professional, and supportive tone. If the user's question is unclear, ask for clarification. If the question is outside the scope of fitness, bodybuilding, or nutrition, politely decline and redirect to relevant topics. If the user asks about previous questions or chat history, use the conversation context to answer as accurately as possible. Always include a 'Sources' section, listing the document, link, or stating 'No knowledge base source used for this answer' if applicable. Never provide medical advice; always recommend consulting a healthcare professional for medical concerns.\n"
        "Safety Notice: Remind users to consult healthcare providers before starting new exercise or nutrition programs, especially if they have pre-existing conditions.\n"
        "\nPrevious User Questions (in order):\n{formatted_history}\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{question}")
    ])

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=prompt,
        return_source_documents=True,
        max_tokens_limit=max_tokens
    )
    return chain
