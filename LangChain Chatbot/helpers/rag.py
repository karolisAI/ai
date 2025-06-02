from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from data.extraction import extract_text_from_pdfs
from langchain.document_loaders import WebBaseLoader
from helpers.baseretriever import TokenLimitedRetriever
import os
from dotenv import load_dotenv
import tiktoken

load_dotenv()
OPENAI_API = os.getenv("OPENAI_API_KEY")

def num_tokens_from_string(string: str, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))

def build_web_vectorstore(urls, embeddings, save_path=None):
    loader = WebBaseLoader(urls)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embeddings)
    if save_path:
        vectorstore.save_local(save_path)
    return vectorstore

def get_combined_retriever(pdf_dir, urls, embedding_store="combined_faiss_index", max_tokens=500):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    
    if not os.path.exists(embedding_store):
        # PDF embeddings
        pdf_texts = extract_text_from_pdfs(pdf_dir)
        pdf_vectorstore = create_embeddings_from_texts(pdf_texts, embeddings)
        
        # web embeddings
        web_vectorstore = build_web_vectorstore(urls, embeddings)
        
        # merge both vectorstores
        pdf_vectorstore.merge_from(web_vectorstore)
        pdf_vectorstore.save_local(embedding_store)
        
    else:
        pdf_vectorstore = FAISS.load_local(embedding_store, embeddings, allow_dangerous_deserialization=True)

    # token-limited retrieval wrapper
    base_retriever = pdf_vectorstore.as_retriever(search_kwargs={"k": 1})

    return TokenLimitedRetriever(base_retriever=base_retriever, max_tokens=max_tokens)

def create_embeddings_from_texts(texts, embeddings, save_path=None):
    documents = []
    for txt in texts:
        documents.append({"page_content": txt['content'], "metadata": {"source": txt['file_name']}})

    vectorstore = FAISS.from_texts(
        texts=[doc['page_content'] for doc in documents],
        embedding=embeddings,
        metadatas=[doc['metadata'] for doc in documents]
    )
    if save_path:
        vectorstore.save_local(save_path)
    return vectorstore

