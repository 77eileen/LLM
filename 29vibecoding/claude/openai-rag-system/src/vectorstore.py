"""
Vector Store Manager for RAG system
FAISS 벡터 저장소 관리 모듈
"""

from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from embeddings import EmbeddingsManager
from config import VECTORSTORE_PATH, TOP_K_RESULTS


class VectorStoreManager:
    """벡터 저장소 관리 클래스"""
    
    def __init__(
        self, 
        embeddings_manager: Optional[EmbeddingsManager] = None,
        vectorstore_path: Path = VECTORSTORE_PATH
    ):
        """
        벡터 저장소 매니저 초기화
        
        Args:
            embeddings_manager: 임베딩 매니저 인스턴스
            vectorstore_path: 벡터 저장소 저장 경로
        """
        self.embeddings_manager = embeddings_manager or EmbeddingsManager()
        self.vectorstore_path = vectorstore_path
        self.vectorstore: Optional[FAISS] = None
        
        print(f"🗄️  벡터 저장소 매니저 초기화")
        print(f"   저장 경로: {vectorstore_path}")
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        문서로부터 새로운 벡터 저장소 생성
        
        Args:
            documents: Document 객체 리스트
            
        Returns:
            FAISS 벡터 저장소
        """
        if not documents:
            raise ValueError("문서가 비어있습니다.")
        
        print(f"\n🔨 벡터 저장소 생성 중... ({len(documents)}개 청크)")
        
        try:
            # FAISS 벡터 저장소 생성
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings_manager.get_embeddings_instance()
            )
            
            print(f"✅ 벡터 저장소 생성 완료")
            return self.vectorstore
            
        except Exception as e:
            print(f"❌ 벡터 저장소 생성 실패: {str(e)}")
            raise
    
    def save_vectorstore(self, folder_path: Optional[Path] = None):
        """
        벡터 저장소를 디스크에 저장
        
        Args:
            folder_path: 저장 경로 (기본값: self.vectorstore_path)
        """
        if self.vectorstore is None:
            raise ValueError("저장할 벡터 저장소가 없습니다.")
        
        save_path = folder_path or self.vectorstore_path
        
        try:
            print(f"\n💾 벡터 저장소 저장 중: {save_path}")
            self.vectorstore.save_local(str(save_path))
            print(f"✅ 저장 완료")
            
        except Exception as e:
            print(f"❌ 저장 실패: {str(e)}")
            raise
    
    def load_vectorstore(self, folder_path: Optional[Path] = None) -> FAISS:
        """
        디스크에서 벡터 저장소 로드
        
        Args:
            folder_path: 로드 경로 (기본값: self.vectorstore_path)
            
        Returns:
            FAISS 벡터 저장소
        """
        load_path = folder_path or self.vectorstore_path
        
        if not load_path.exists():
            raise FileNotFoundError(
                f"벡터 저장소를 찾을 수 없습니다: {load_path}\n"
                "먼저 문서를 인덱싱해주세요."
            )
        
        try:
            print(f"\n📂 벡터 저장소 로드 중: {load_path}")
            
            self.vectorstore = FAISS.load_local(
                folder_path=str(load_path),
                embeddings=self.embeddings_manager.get_embeddings_instance(),
                allow_dangerous_deserialization=True  # FAISS 로드 시 필요
            )
            
            print(f"✅ 로드 완료")
            return self.vectorstore
            
        except Exception as e:
            print(f"❌ 로드 실패: {str(e)}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = TOP_K_RESULTS
    ) -> List[Document]:
        """
        쿼리와 유사한 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 개수
            
        Returns:
            유사한 Document 리스트
        """
        if self.vectorstore is None:
            raise ValueError("벡터 저장소가 로드되지 않았습니다.")
        
        try:
            print(f"\n🔍 유사 문서 검색 중... (Top-{k})")
            results = self.vectorstore.similarity_search(query, k=k)
            print(f"✅ {len(results)}개 문서 검색 완료")
            return results
            
        except Exception as e:
            print(f"❌ 검색 실패: {str(e)}")
            raise
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = TOP_K_RESULTS
    ) -> List[tuple[Document, float]]:
        """
        쿼리와 유사한 문서를 유사도 점수와 함께 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 개수
            
        Returns:
            (Document, 유사도 점수) 튜플 리스트
        """
        if self.vectorstore is None:
            raise ValueError("벡터 저장소가 로드되지 않았습니다.")
        
        try:
            print(f"\n🔍 유사 문서 검색 중... (Top-{k}, 점수 포함)")
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            print(f"✅ {len(results)}개 문서 검색 완료")
            return results
            
        except Exception as e:
            print(f"❌ 검색 실패: {str(e)}")
            raise
    
    def get_retriever(self, k: int = TOP_K_RESULTS):
        """
        LangChain Retriever 반환 (RAG Chain에서 사용)
        
        Args:
            k: 반환할 문서 개수
            
        Returns:
            VectorStoreRetriever
        """
        if self.vectorstore is None:
            raise ValueError("벡터 저장소가 로드되지 않았습니다.")
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def vectorstore_exists(self, folder_path: Optional[Path] = None) -> bool:
        """
        벡터 저장소 파일 존재 여부 확인
        
        Args:
            folder_path: 확인할 경로
            
        Returns:
            존재 여부
        """
        check_path = folder_path or self.vectorstore_path
        return check_path.exists() and (check_path / "index.faiss").exists()


def test_vectorstore():
    """벡터 저장소 테스트"""
    print("=== Vector Store Manager 테스트 ===\n")
    
    from document_loader import DocumentLoader
    
    try:
        # 문서 로더로 문서 로드
        print("1️⃣ 문서 로딩...")
        loader = DocumentLoader()
        chunks = loader.load_and_split()
        
        if not chunks:
            print("⚠️  문서가 없어 테스트를 건너뜁니다.")
            return
        
        # 벡터 저장소 생성
        print("\n2️⃣ 벡터 저장소 생성...")
        vs_manager = VectorStoreManager()
        vs_manager.create_vectorstore(chunks)
        
        # 저장
        print("\n3️⃣ 벡터 저장소 저장...")
        vs_manager.save_vectorstore()
        
        # 로드 테스트
        print("\n4️⃣ 벡터 저장소 로드 테스트...")
        vs_manager2 = VectorStoreManager()
        vs_manager2.load_vectorstore()
        
        # 검색 테스트
        print("\n5️⃣ 유사도 검색 테스트...")
        query = "문서의 주요 내용은 무엇인가요?"
        results = vs_manager2.similarity_search(query, k=3)
        
        print(f"\n--- 검색 결과 ---")
        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] 소스: {doc.metadata.get('source', 'Unknown')}")
            print(f"내용: {doc.page_content[:150]}...")
        
        print(f"\n✅ 벡터 저장소 테스트 성공!")
        
    except FileNotFoundError as e:
        print(f"⚠️  {str(e)}")
        print("data/ 폴더에 문서를 추가한 후 다시 시도하세요.")
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")


if __name__ == "__main__":
    test_vectorstore()
