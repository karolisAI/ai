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

load_dotenv()
OPENAI_API = os.getenv("OPENAI_API_KEY")

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
    
    try:
        # Get embedding dimension from a sample
        sample_text = texts[0]
        sample_embedding = embedder.embed_query(sample_text)
        expected_dim = len(sample_embedding)
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            current_batch = (i // batch_size) + 1
            
            # Ensure batch doesn't exceed maximum size
            if len(batch) > MAX_BATCH_SIZE:
                logging.warning(f"Batch size {len(batch)} exceeds maximum {MAX_BATCH_SIZE}, truncating")
                batch = batch[:MAX_BATCH_SIZE]
            
            for attempt in range(max_retries):
                try:
                    logging.info(f"Processing batch {current_batch}/{total_batches} (size: {len(batch)})")
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
                            error=e
                        )
                        logging.error(f"Failed to embed batch {current_batch} after {max_retries} attempts: {str(e)}")
                        raise
                    logging.warning(f"Attempt {attempt + 1} failed for batch {current_batch}, retrying...")
                    time.sleep(1)  # Add delay between retries
        
        # Validate final results
        validate_embeddings(results, expected_dim)
        return results
        
    except Exception as e:
        log_embedding_operation(
            "batch_embed",
            batch_size,
            len(texts),
            success=False,
            error=e
        )
        raise

class OpenAIEmbeddingFunction(EmbeddingFunction):
    """Wrapper for OpenAI embeddings to match ChromaDB's interface."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API)
    
    def __call__(self, input: Documents) -> List[List[float]]:
        if isinstance(input, str):
            input = [input]
        # Use a smaller batch size for ChromaDB interface to avoid exceeding limits
        return batch_embed(input, self.embeddings, batch_size=50)

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
    loaded_pdfs = []
    if pdf_dir and os.path.exists(pdf_dir):
        pdf_loader = DirectoryLoader(pdf_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        loaded_pdfs = [doc.metadata.get('source', 'unknown') for doc in pdf_docs]
        logging.info(f"[RAG] Loaded PDFs: {loaded_pdfs}")
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
    def __init__(self, api_key):
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return [self.embeddings.embed_query(text) for text in input]

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
