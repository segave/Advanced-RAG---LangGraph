import pytest
import os
import tempfile
from pathlib import Path
import shutil
from backend.document_processor import (
    DocumentIngester,
    TikTokenTextSplitter,
    ChromaVectorStore,
    PDFLoader,
    FileLoader
)

@pytest.fixture(scope="function")
def vector_store_path():
    """Create a temporary directory for the vector store"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up - force removal after small delay
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        import time
        time.sleep(1)  # Give OS time to release files
        shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sample_files():
    """Create temporary test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a sample PDF file with valid content
        pdf_path = Path(tmp_dir) / "test.pdf"
        with open(pdf_path, "wb") as f:
            # Create a minimal valid PDF
            f.write(b"""
%PDF-1.0
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]>>endobj
xref
0 4
0000000000 65535 f
0000000010 00000 n
0000000053 00000 n
0000000102 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
149
%%EOF
""")
            
        # Create a sample TXT file
        txt_path = Path(tmp_dir) / "test.txt"
        with open(txt_path, "w", encoding='utf-8') as f:
            f.write("This is a test document")
            
        # Create a sample DOCX file
        docx_path = Path(tmp_dir) / "test.docx"
        with open(docx_path, "w", encoding='utf-8') as f:
            f.write("This is a test DOCX document")
        
        yield {
            "pdf": str(pdf_path),
            "txt": str(txt_path),
            "docx": str(docx_path),
            "temp_dir": tmp_dir
        }

@pytest.fixture
def document_processor(vector_store_path):
    """Create a document processor instance"""
    text_splitter = TikTokenTextSplitter(chunk_size=250, chunk_overlap=0)
    vector_store = ChromaVectorStore(
        collection_name="test-collection",
        persist_directory=vector_store_path
    )
    processor = DocumentIngester(text_splitter, vector_store)
    yield processor
    # Cleanup
    del vector_store

def test_txt_ingestion(sample_files, document_processor):
    """Test TXT document ingestion"""
    loader = FileLoader([sample_files["txt"]])
    try:
        result = document_processor.process_documents(loader)
        assert result is not None
    except Exception as e:
        pytest.fail(f"TXT ingestion failed: {str(e)}")

def test_docx_ingestion(sample_files, document_processor):
    """Test DOCX document ingestion"""
    loader = FileLoader([sample_files["docx"]])
    try:
        result = document_processor.process_documents(loader)
        assert result is not None
    except Exception as e:
        pytest.fail(f"DOCX ingestion failed: {str(e)}")

def test_mixed_document_ingestion(sample_files, document_processor):
    """Test ingestion of multiple document types"""
    loader = FileLoader([
        sample_files["txt"],
        sample_files["docx"]
    ])
    try:
        result = document_processor.process_documents(loader)
        assert result is not None
    except Exception as e:
        pytest.fail(f"Mixed document ingestion failed: {str(e)}")

def test_invalid_file_path(vector_store_path):
    """Test handling of invalid file paths"""
    loader = FileLoader(["nonexistent.txt"])
    text_splitter = TikTokenTextSplitter()
    vector_store = ChromaVectorStore(
        collection_name="test-collection",
        persist_directory=vector_store_path
    )
    processor = DocumentIngester(text_splitter, vector_store)
    
    with pytest.raises(Exception):
        processor.process_documents(loader) 