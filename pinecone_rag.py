# core/pinecone_rag.py
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List, Optional

class PineconeRAG:
    """Handles all Pinecone RAG operations for the AI Lab"""
    
    def __init__(
        self, 
        index_name: str,
        openai_api_key: str,
        pinecone_api_key: str,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize Pinecone RAG system
        
        Args:
            index_name: Name of Pinecone index
            openai_api_key: OpenAI API key for embeddings
            pinecone_api_key: Pinecone API key
            embedding_model: OpenAI embedding model to use
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key
        )
        self.index_name = index_name
        self.pinecone_api_key = pinecone_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def query(
        self, 
        question: str, 
        k: int = 3,
        filter: Optional[dict] = None
    ) -> str:
        """
        Query the Pinecone index for relevant information
        
        Args:
            question: Query string
            k: Number of documents to retrieve
            filter: Optional metadata filter
            
        Returns:
            Concatenated relevant documents
        """
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        
        docs = vectorstore.similarity_search(
            question, 
            k=k,
            filter=filter
        )
        
        return self._format_results(docs)
    
    def add_documents(self, text: str, metadata: dict = None) -> None:
        """
        Add documents to Pinecone index
        
        Args:
            text: Text content to add
            metadata: Optional metadata dictionary
        """
        from langchain.docstore.document import Document
        
        if metadata is None:
            metadata = {}
            
        documents = [Document(page_content=text, metadata=metadata)]
        split_docs = self.text_splitter.split_documents(documents)
        
        PineconeVectorStore.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            index_name=self.index_name
        )
    
    def _format_results(self, docs: List) -> str:
        """Format retrieved documents with citations"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown source')
            formatted.append(
                f"DOCUMENT {i} [Source: {source}]:\n"
                f"{doc.page_content}\n"
                f"{'-'*40}"
            )
        return "\n\n".join(formatted)