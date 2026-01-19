"""
Simple RAG chatbot using ChromaDB + Google Gemini.
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def initialize_chatbot():
    """Initialize the RAG chatbot with ChromaDB and Gemini."""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "":
        raise ValueError("GOOGLE_API_KEY is not set. Please add it to your .env file.")
    
    print("🔧 Initializing chatbot...")
    
    # Load embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Load existing ChromaDB
    vectorstore = Chroma(
        collection_name="my_rag_db",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"✅ Loaded ChromaDB from ./chroma_db")
    
    # Initialize Gemini LLM (use lite model for better availability)
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-flash-lite-latest",
        temperature=0.7,
    )
    print(f"✅ Initialized Gemini LLM (models/gemini-flash-lite-latest)")
    
    # Create retriever (smaller k to reduce token usage)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Create RAG prompt template
    template = """다음 문서를 참고하여 질문에 답변하세요. 문서에 없는 내용은 "문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.

참고 문서:
{context}

질문: {question}

답변:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create RAG chain
    def format_docs(docs):
        return "\n\n".join([f"[문서 {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ Chatbot ready!\n")
    return rag_chain, retriever


def chat_loop(rag_chain, retriever):
    """Run interactive chat loop."""
    print("=" * 70)
    print("💬 RAG Chatbot (문서 기반 질문답변)")
    print("=" * 70)
    print("📌 업로드된 문서를 기반으로 질문에 답변합니다.")
    print("📌 종료하려면 'exit', 'quit', '종료' 중 하나를 입력하세요.\n")
    
    while True:
        # Get user input
        user_input = input("🙋 질문: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ["exit", "quit", "종료", "나가기"]:
            print("\n👋 채팅을 종료합니다. 안녕히 가세요!")
            break
        
        if not user_input:
            print("⚠️  질문을 입력해주세요.\n")
            continue
        
        # Get response from chatbot
        print("\n🤖 답변 생성 중...\n")
        try:
            # Get answer
            answer = rag_chain.invoke(user_input)
            
            # Get source documents
            source_docs = retriever.invoke(user_input)
            
            # Print answer
            print("=" * 70)
            print(f"💡 답변:\n{answer}")
            print("=" * 70)
            
            # Print sources (optional)
            if source_docs:
                print(f"\n📚 참고한 문서 ({len(source_docs)}개):")
                for i, doc in enumerate(source_docs, 1):
                    preview = doc.page_content[:150].replace("\n", " ")
                    print(f"  [{i}] {preview}...")
            
            print("\n")
            
        except Exception as e:
            # Fallback: if quota/rate limit hit, return top retrieved chunks as answer
            err_msg = str(e)
            print(f"❌ 오류 발생: {err_msg}\n")
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                try:
                    source_docs = retriever.invoke(user_input)
                    if source_docs:
                        joined = "\n\n".join([doc.page_content for doc in source_docs])
                        print("=" * 70)
                        print("💡 LLM 사용 제한으로 검색 결과를 직접 반환합니다:")
                        print(joined[:1500])
                        print("=" * 70)
                        print(f"\n📚 참고한 문서 ({len(source_docs)}개):")
                        for i, doc in enumerate(source_docs, 1):
                            preview = doc.page_content[:150].replace("\n", " ")
                            print(f"  [{i}] {preview}...")
                        print("\n")
                    else:
                        print("⚠️ 검색 결과가 없습니다. 질문을 더 구체적으로 입력해보세요.\n")
                except Exception as e2:
                    print(f"❌ 대체 경로도 실패: {e2}\n")


def main():
    """Main entry point."""
    try:
        rag_chain, retriever = initialize_chatbot()
        chat_loop(rag_chain, retriever)
    except Exception as e:
        print(f"\n❌ 초기화 실패: {e}")
        print("💡 Tip: ./chroma_db가 존재하는지, GOOGLE_API_KEY가 설정되어 있는지 확인하세요.")


if __name__ == "__main__":
    main()
