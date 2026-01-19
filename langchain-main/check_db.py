"""
Quick script to inspect ChromaDB contents.
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not set")

# Load existing ChromaDB
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = Chroma(
    collection_name="my_rag_db",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Get basic info
all_data = vectorstore._collection.get(include=["documents", "metadatas"])
print(f"✅ ChromaDB loaded successfully!")
print(f"📊 Total documents stored: {len(all_data['ids'])}")

# Show first document sample
if all_data["documents"]:
    print(f"\n📄 Sample document (first 300 chars):")
    print(all_data["documents"][0][:300])
    print("...")

# Test similarity search
print("\n🔍 Testing similarity search...")
test_query = "조선의 역사에 대해 알려줘"
results = vectorstore.similarity_search_with_score(test_query, k=3)

print(f"\nQuery: '{test_query}'")
print(f"Top {len(results)} results:\n")
for i, (doc, score) in enumerate(results, 1):
    print(f"[{i}] Distance score: {score:.4f}")
    print(f"Content preview: {doc.page_content[:200]}...\n")

print("✅ ChromaDB is working correctly!")
