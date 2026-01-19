"""
Document ingestion pipeline for RAG system.
Loads .txt files from ./data, splits into chunks, embeds, and stores in ChromaDB.
"""

import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


def validate_env():
    """Validate that GOOGLE_API_KEY is set in environment."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "":
        raise ValueError(
            "GOOGLE_API_KEY is not set. Please add it to your .env file (Google AI Studio)."
        )
    print("✓ GOOGLE_API_KEY loaded successfully")


def load_documents():
    """Load all .txt files from ./data directory."""
    data_path = "./data"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data directory '{data_path}' does not exist. Please create it and add .txt files."
        )
    
    # Load documents with auto-detection for Korean encoding
    loader = DirectoryLoader(
        path=data_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True}
    )
    
    documents = loader.load()
    
    if not documents:
        raise ValueError(
            f"No .txt files found in '{data_path}'. Please add documents to ingest."
        )
    
    print(f"✓ Loaded {len(documents)} documents from {data_path}")
    return documents


def split_documents(documents):
    """Split documents into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Split into {len(chunks)} chunks")
    return chunks


def rebuild_vector_db(chunks):
    """Create and persist ChromaDB vector store."""
    persist_directory = "./chroma_db"
    collection_name = "my_rag_db"
    
    # Delete existing DB if it exists (with retry)
    if os.path.exists(persist_directory):
        print(f"⚠ Removing existing database at {persist_directory}")
        import time
        import stat
        
        def handle_remove_error(func, path, exc_info):
            os.chmod(path, stat.S_IWRITE)
            time.sleep(0.1)
            func(path)
        
        try:
            shutil.rmtree(persist_directory, onerror=handle_remove_error)
        except Exception as e:
            print(f"⚠ Failed to remove DB (may be in use): {e}")
            print("💡 Tip: Close Jupyter kernel and try again, or manually delete ./chroma_db")
            raise
    
    # Initialize embeddings (Gemini via Google AI Studio)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    print("✓ Initialized Google Generative AI embeddings")
    
    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    print(f"✓ Created ChromaDB at {persist_directory} (collection: {collection_name})")
    return vectorstore


def main():
    """Main ingestion pipeline."""
    print("=" * 70)
    print("Starting document ingestion pipeline...")
    print("=" * 70)
    
    try:
        # Step 1: Validate environment
        validate_env()
        
        # Step 2: Load documents
        documents = load_documents()
        
        # Step 3: Split into chunks
        chunks = split_documents(documents)
        
        # Step 4 & 5: Embed and store in ChromaDB
        rebuild_vector_db(chunks)
        
        # Step 6: Summary
        print("=" * 70)
        print(f"✅ SUCCESS: Loaded {len(documents)} documents, "
              f"split into {len(chunks)} chunks, "
              f"and saved to ChromaDB at ./chroma_db (collection: my_rag_db)")
        print("=" * 70)
        
    except Exception as e:
        print("=" * 70)
        print(f"❌ ERROR: {str(e)}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    main()
