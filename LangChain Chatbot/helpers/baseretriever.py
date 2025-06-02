from langchain.schema import BaseRetriever, Document
from typing import List
import tiktoken
from langchain.vectorstores import FAISS
from pydantic import Field

class TokenLimitedRetriever(BaseRetriever):
    base_retriever: BaseRetriever = Field(...)
    max_tokens: int = Field(default=300)
    model_name: str = Field(default="gpt-3.5-turbo")

    def num_tokens(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model_name)
        return len(encoding.encode(text))

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.get_relevant_documents(query)
        final_docs = []
        current_tokens = 0

        for doc in docs:
            doc_tokens = self.num_tokens(doc.page_content)
            if current_tokens + doc_tokens <= self.max_tokens:
                final_docs.append(doc)
                current_tokens += doc_tokens
            else:
                remaining_tokens = self.max_tokens - current_tokens
                if remaining_tokens > 0:
                    encoding = tiktoken.encoding_for_model(self.model_name)
                    encoded_doc = encoding.encode(doc.page_content)[:remaining_tokens]
                    truncated_content = encoding.decode(encoded_doc)
                    doc.page_content = truncated_content
                    final_docs.append(doc)
                break

        return final_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)