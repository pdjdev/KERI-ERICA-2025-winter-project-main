# LLM Project with LangChain

## 프로젝트 개요
사내 문서 기반 RAG(Retrieval-Augmented Generation) 챗봇입니다. 업로드된 문서를 기반으로 질문에 답변하며, 웹 검색을 결합하여 더 풍부한 정보를 제공합니다.

## 기술 스택
- **LangChain**: LLM 애플리케이션 프레임워크
- **OpenAI**: 언어 모델 (GPT)
- **ChromaDB**: 벡터 데이터베이스
- **Google Search**: 웹 검색 도구
- **Interface**: CLI (커맨드 라인)

## 현재 상태
⚠️ 이 프로젝트는 현재 스캐폴딩 단계입니다. 실제 구현 로직은 아직 작성되지 않았습니다.

## 설치
```bash
pip install -r requirements.txt
```

## 환경 변수 설정
`.env` 파일에 필요한 API 키를 설정하세요:
- `OPENAI_API_KEY`: OpenAI API 키
- `GOOGLE_API_KEY`: Google Search API 키
