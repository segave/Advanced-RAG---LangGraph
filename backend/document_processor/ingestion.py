from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from langchain_openai import OpenAIEmbeddings
from typing import List, Any, Union
import os
from .interfaces import DocumentLoader, TextSplitter, VectorStore
import docx
import psutil
import shutil

load_dotenv()

class WebLoader(DocumentLoader):
    def __init__(self, urls: List[str]):
        self.urls = urls

    def load(self) -> List[Any]:
        docs = [WebBaseLoader(url).load() for url in self.urls]
        return [item for sublist in docs for item in sublist]

class PDFLoader(DocumentLoader):
    def __init__(self, pdf_files: List[str]):
        self.pdf_files = pdf_files

    def load(self) -> List[Any]:
        docs = []
        for pdf in self.pdf_files:
            if os.path.exists(pdf):
                loader = PyPDFLoader(pdf)
                docs.extend(loader.load())
        return docs

class FileLoader(DocumentLoader):
    def __init__(self, text_files: List[str]):
        self.text_files = text_files

    def load(self) -> List[Any]:
        docs = []
        for text_file in self.text_files:
            if os.path.exists(text_file):
                loader = TextLoader(text_file)
                docs.extend(loader.load())
        return docs

class DirectoryDocumentLoader(DocumentLoader):
    def __init__(self, directory_path: str, glob_pattern: str = "**/*"):
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern

    def load(self) -> List[Any]:
        loader = DirectoryLoader(
            self.directory_path,
            glob=self.glob_pattern,
            use_multithreading=True
        )
        return loader.load()

class DocxLoader(DocumentLoader):
    """Custom loader for DOCX files"""
    def __init__(self, docx_files: List[str]):
        self.docx_files = docx_files

    def load(self) -> List[Any]:
        from langchain_core.documents import Document
        docs = []
        
        for file_path in self.docx_files:
            if os.path.exists(file_path):
                doc = docx.Document(file_path)
                # Extract text from paragraphs
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                docs.append(Document(
                    page_content=text,
                    metadata={"source": file_path}
                ))
        return docs

class RecursiveTextSplitter(TextSplitter):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", ",", " ", ""],
            length_function=len
        )

    def split_documents(self, documents: List[Any]) -> List[Any]:
        return self.splitter.split_documents(documents)

class ChromaVectorStore(VectorStore):
    def __init__(self, collection_name: str = "rag-chroma", persist_directory: str = "./.chroma"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = OpenAIEmbeddings()
        self.vectorstore = None
        self._client = None

    def _get_client(self):
        """Create a new client instance"""
        from chromadb import Client
        from chromadb.config import Settings
        
        if not self._client:
            self._client = Client(Settings(
                persist_directory=self.persist_directory,
                is_persistent=True
            ))
        return self._client

    def store_documents(self, documents: List[Any]) -> Any:
        client = self._get_client()

        # Create and return the vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            collection_name=self.collection_name,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory,
            client=client
        )
        return self.vectorstore

    def get_retriever(self) -> Any:
        if not self.vectorstore:
            client = self._get_client()

            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                client=client
            )
        return self.vectorstore.as_retriever()

    def cleanup(self):
        """Clean up vector store directory"""
        try:
            # First, try to delete the collection
            if self.vectorstore is not None and hasattr(self.vectorstore, '_collection'):
                # Delete all documents using a valid where clause
                self.vectorstore._collection.delete(where={"source": {"$ne": ""}})

            # Then wait a bit
            import time
            time.sleep(1)

            # Finally remove the directory
            if os.path.exists(self.persist_directory):
                # Get current process
                current_process = psutil.Process()
                
                # Close all file handles in the directory
                for handler in current_process.open_files():
                    if self.persist_directory in handler.path:
                        try:
                            os.close(handler.fd)
                        except:
                            pass

                # Remove directory
                shutil.rmtree(self.persist_directory, ignore_errors=True)
                print(f"Successfully cleaned up vector store at {self.persist_directory}")
        except Exception as e:
            print(f"Error cleaning up vector store: {str(e)}")

class DocumentIngester:
    def __init__(
        self,
        text_splitter: TextSplitter,
        vector_store: VectorStore
    ):
        self.text_splitter = text_splitter
        self.vector_store = vector_store

    def process_documents(self, document_loader: DocumentLoader) -> Any:
        """Procesa y almacena los documentos"""
        documents = document_loader.load()
        split_docs = self.text_splitter.split_documents(documents)
        return self.vector_store.store_documents(split_docs)

# Ejemplo de uso:
if __name__ == "__main__":
    # Configurar componentes
    text_splitter = RecursiveTextSplitter()
    vector_store = ChromaVectorStore()
    ingester = DocumentIngester(text_splitter, vector_store)
    
    # Ejemplo con URLs
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    ]
    web_loader = WebLoader(urls)
    
    # Procesar documentos
    ingester.process_documents(web_loader)

def get_document_loader(file_paths: list[str]) -> Union[FileLoader, PDFLoader, DocxLoader]:
    """Determines the appropriate loader based on file extensions"""
    # Group files by extension
    pdf_files = [f for f in file_paths if f.lower().endswith('.pdf')]
    docx_files = [f for f in file_paths if f.lower().endswith('.docx')]
    txt_files = [f for f in file_paths if f.lower().endswith('.txt')]
    
    if pdf_files and not (docx_files or txt_files):
        return PDFLoader(pdf_files)
    elif docx_files and not (pdf_files or txt_files):
        return DocxLoader(docx_files)
    elif txt_files and not (pdf_files or docx_files):
        return FileLoader(txt_files)
    else:
        # Si hay una mezcla de tipos, usamos DirectoryLoader
        directory = os.path.dirname(file_paths[0])
        return DirectoryDocumentLoader(directory)