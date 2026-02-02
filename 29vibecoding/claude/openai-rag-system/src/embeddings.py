"""
Embeddings Manager for RAG system
OpenAI 임베딩 생성 및 관리 모듈
"""

from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config import OPENAI_API_KEY, EMBEDDING_MODEL


class EmbeddingsManager:
    """임베딩 생성 및 관리 클래스"""
    
    def __init__(self, model: str = EMBEDDING_MODEL):
        """
        임베딩 매니저 초기화
        
        Args:
            model: OpenAI 임베딩 모델 이름
        """
        self.model = model
        self.embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=OPENAI_API_KEY
        )
        print(f"🔧 임베딩 모델 초기화: {model}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 문서를 임베딩으로 변환
        
        Args:
            texts: 문서 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
        """
        try:
            print(f"🔄 {len(texts)}개 문서 임베딩 생성 중...")
            embeddings = self.embeddings.embed_documents(texts)
            print(f"✅ 임베딩 생성 완료 (차원: {len(embeddings[0])})")
            return embeddings
        except Exception as e:
            print(f"❌ 임베딩 생성 실패: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리를 임베딩으로 변환
        
        Args:
            text: 쿼리 텍스트
            
        Returns:
            임베딩 벡터
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            print(f"❌ 쿼리 임베딩 실패: {str(e)}")
            raise
    
    def get_embeddings_instance(self):
        """
        LangChain Embeddings 인스턴스 반환
        (VectorStore 생성 시 사용)
        
        Returns:
            OpenAIEmbeddings 인스턴스
        """
        return self.embeddings
    
    def get_model_info(self) -> dict:
        """
        모델 정보 반환
        
        Returns:
            모델 정보 딕셔너리
        """
        return {
            "model": self.model,
            "provider": "OpenAI",
            "embedding_dimension": self._get_embedding_dimension()
        }
    
    def _get_embedding_dimension(self) -> int:
        """임베딩 차원 수 반환"""
        # text-embedding-3-small: 1536 차원
        # text-embedding-3-large: 3072 차원
        if "small" in self.model:
            return 1536
        elif "large" in self.model:
            return 3072
        else:
            return 1536  # 기본값


def test_embeddings():
    """임베딩 매니저 테스트"""
    print("=== Embeddings Manager 테스트 ===\n")
    
    try:
        # 임베딩 매니저 생성
        embeddings_manager = EmbeddingsManager()
        
        # 모델 정보 출력
        info = embeddings_manager.get_model_info()
        print(f"\n📊 모델 정보:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # 테스트 텍스트
        test_texts = [
            "인공지능은 컴퓨터 과학의 한 분야입니다.",
            "머신러닝은 인공지능의 하위 분야입니다.",
            "딥러닝은 머신러닝의 한 방법입니다."
        ]
        
        print(f"\n🧪 테스트 텍스트 {len(test_texts)}개 임베딩 생성...")
        
        # 문서 임베딩
        doc_embeddings = embeddings_manager.embed_documents(test_texts)
        print(f"   문서 임베딩 차원: {len(doc_embeddings[0])}")
        print(f"   첫 5개 값: {doc_embeddings[0][:5]}")
        
        # 쿼리 임베딩
        query = "인공지능에 대해 알려주세요"
        print(f"\n🔍 쿼리 임베딩 생성: '{query}'")
        query_embedding = embeddings_manager.embed_query(query)
        print(f"   쿼리 임베딩 차원: {len(query_embedding)}")
        print(f"   첫 5개 값: {query_embedding[:5]}")
        
        print(f"\n✅ 임베딩 테스트 성공!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")


if __name__ == "__main__":
    test_embeddings()
