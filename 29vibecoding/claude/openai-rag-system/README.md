# OpenAI + RAG System

최신 라이브러리를 사용한 문서 기반 질의응답 시스템

## 프로젝트 구조

```
openai-rag-system/
├── .env                    # 환경 변수 (API 키)
├── .env.example            # 환경 변수 예시
├── requirements.txt        # Python 패키지
├── data/                   # 문서 파일 저장 (.doc, .docx)
├── vectorstore/            # 벡터 DB 저장
├── src/
│   ├── __init__.py
│   ├── config.py           # 설정 관리
│   ├── document_loader.py  # 문서 로딩
│   ├── embeddings.py       # 임베딩 생성
│   ├── vectorstore.py      # 벡터 저장소 관리
│   ├── rag_chain.py        # RAG 체인
│   └── main.py             # 메인 실행
└── README.md
```

## 설치 방법

### 1. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정
`.env` 파일에 OpenAI API 키를 입력하세요:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 4. 문서 추가
`data/` 폴더에 `.doc` 또는 `.docx` 파일을 추가하세요.

## 사용 방법

### 1단계: 문서 인덱싱
```bash
python src/main.py --index
```

### 2단계: 질문하기
```bash
python src/main.py --query "당신의 질문을 입력하세요"
```

### 대화형 모드
```bash
python src/main.py --interactive
```

## 기능

- ✅ `.doc`, `.docx` 문서 자동 로딩
- ✅ OpenAI Embeddings (text-embedding-3-small)
- ✅ FAISS 벡터 저장소
- ✅ 문서 기반 질의응답
- ✅ 소스 문서 참조 표시
- ✅ 대화형 인터페이스

## 기술 스택

- **LangChain**: RAG 파이프라인
- **OpenAI**: GPT-4o-mini, text-embedding-3-small
- **FAISS**: 벡터 저장소
- **python-docx**: 문서 처리

## 라이브러리 버전

모든 라이브러리는 2024-2025 최신 stable 버전을 사용합니다.
- Deprecated 된 기능 없음
- 호환성 충돌 해결됨

## 트러블슈팅

### API 키 오류
`.env` 파일이 제대로 설정되었는지 확인하세요.

### 문서 로딩 오류
`data/` 폴더에 `.doc` 또는 `.docx` 파일이 있는지 확인하세요.

### 벡터 저장소 오류
`vectorstore/` 폴더를 삭제하고 다시 인덱싱하세요.

## 라이선스

MIT License
