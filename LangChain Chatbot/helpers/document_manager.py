import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import streamlit as st
from datetime import datetime

class DocumentManager:
    def __init__(self, upload_dir: str = "./data/uploads"):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)
        
    def save_uploaded_file(self, uploaded_file) -> Optional[str]:
        """Save an uploaded file and return its path"""
        try:
            # Create a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{uploaded_file.name}"
            file_path = os.path.join(self.upload_dir, filename)
            
            # Save the file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            return file_path
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
            return None
    
    def process_document(self, file_path: str) -> List[Document]:
        """Process a document and return its chunks"""
        try:
            # Determine file type and load accordingly
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            # Load and split the document
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Add metadata
            chunks = splitter.split_documents(documents)
            for chunk in chunks:
                chunk.metadata.update({
                    "source_type": "user_upload",
                    "file_name": os.path.basename(file_path),
                    "upload_date": datetime.now().isoformat()
                })
            
            return chunks
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return []
    
    def get_uploaded_documents(self) -> List[str]:
        """Get list of uploaded documents"""
        try:
            return [f for f in os.listdir(self.upload_dir) if os.path.isfile(os.path.join(self.upload_dir, f))]
        except Exception as e:
            st.error(f"Error listing documents: {str(e)}")
            return []
    
    def delete_document(self, filename: str) -> bool:
        """Delete an uploaded document"""
        try:
            file_path = os.path.join(self.upload_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            st.error(f"Error deleting document: {str(e)}")
            return False 