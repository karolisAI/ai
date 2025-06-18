from typing import List
from langchain.schema import BaseRetriever, Document
from langchain_community.retrievers import BM25Retriever
import tiktoken

class EnhancedRetriever(BaseRetriever):
    """Enhanced retriever with hybrid search and memory optimization."""
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        max_tokens: int = 500,
        cache_size: int = 100,
        hybrid_search_weight: float = 0.7
    ):
        """Initialize with base retriever and configuration."""
        super().__init__()
        self._base_retriever = base_retriever
        self._max_tokens = max_tokens
        self._encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self._hybrid_search_weight = hybrid_search_weight
        # No document re-embedding here!
        self._bm25_retriever = None
        self._documents = []

    def add_documents(self, documents: List[Document]):
        """Add documents to BM25 retriever for keyword search."""
        self._documents.extend(documents)
        self._bm25_retriever = BM25Retriever.from_documents(documents)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents using hybrid search and token limiting."""
        # Get semantic search results from Chroma (via self._base_retriever)
        semantic_docs = self._base_retriever.invoke(query)
        # Get keyword search results from BM25
        keyword_docs = self._bm25_retriever.invoke(query) if self._bm25_retriever else []
        # Combine and score documents
        all_docs = set(semantic_docs + keyword_docs)
        scored_docs = []
        for doc in all_docs:
            semantic_score = 1.0 if doc in semantic_docs else 0.0
            keyword_score = 1.0 if doc in keyword_docs else 0.0
            combined_score = (
                self._hybrid_search_weight * semantic_score +
                (1 - self._hybrid_search_weight) * keyword_score
            )
            scored_docs.append((doc, combined_score))
        # Sort by combined score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        # Apply token limit
        total_tokens = 0
        limited_docs = []
        for doc, _ in scored_docs:
            doc_tokens = len(self._encoding.encode(doc.page_content))
            if total_tokens + doc_tokens <= self._max_tokens:
                limited_docs.append(doc)
                total_tokens += doc_tokens
            else:
                break
        return limited_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        return self._get_relevant_documents(query)

    def clear_cache(self):
        """No-op for compatibility."""
        pass