"""
Configuration management for RAG system
환경 변수 및 시스템 설정 관리
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 디렉토리
BASE_DIR = Path(__file__).parent.parent

# OpenAI API 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")

# 모델 설정
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# RAG 설정
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# 디렉토리 설정
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_TYPE = os.getenv("VECTORSTORE_TYPE", "faiss")

# 디렉토리 생성
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

# 벡터 저장소 경로
VECTORSTORE_PATH = VECTORSTORE_DIR / "faiss_index"

# 지원하는 문서 확장자
SUPPORTED_EXTENSIONS = [".doc", ".docx"]

def get_config_info():
    """설정 정보 출력"""
    return {
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "LLM_MODEL": LLM_MODEL,
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "TOP_K_RESULTS": TOP_K_RESULTS,
        "DATA_DIR": str(DATA_DIR),
        "VECTORSTORE_DIR": str(VECTORSTORE_DIR),
        "VECTORSTORE_TYPE": VECTORSTORE_TYPE,
    }

if __name__ == "__main__":
    print("=== RAG System Configuration ===")
    for key, value in get_config_info().items():
        print(f"{key}: {value}")
