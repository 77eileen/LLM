"""
Main entry point for RAG system
RAG 시스템 메인 실행 파일
"""

import argparse
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config_info
from document_loader import DocumentLoader
from vectorstore import VectorStoreManager
from rag_chain import RAGChain


def index_documents():
    """
    문서를 로드하고 벡터 저장소 생성
    """
    print("\n" + "="*60)
    print("📚 문서 인덱싱 시작")
    print("="*60 + "\n")

    try:
        # 1. 문서 로드 및 청킹
        loader = DocumentLoader()
        chunks = loader.load_and_split()

        if not chunks:
            print("❌ 로드할 문서가 없습니다.")
            return False

        # 2. 벡터 저장소 생성
        vs_manager = VectorStoreManager()
        vs_manager.create_vectorstore(chunks)

        # 3. 저장
        vs_manager.save_vectorstore()

        print("\n" + "="*60)
        print("✅ 인덱싱 완료!")
        print("="*60)
        print(f"총 {len(chunks)}개 청크가 벡터 저장소에 저장되었습니다.\n")

        return True

    except FileNotFoundError as e:
        print(f"\n❌ 오류: {str(e)}")
        print("data/ 폴더에 .doc 또는 .docx 파일을 추가해주세요.\n")
        return False
    except Exception as e:
        print(f"\n❌ 인덱싱 실패: {str(e)}\n")
        return False


def query_single(question: str):
    """
    단일 질문에 답변

    Args:
        question: 사용자 질문
    """
    print("\n" + "="*60)
    print("🔍 질의응답 모드")
    print("="*60)

    try:
        # 벡터 저장소 로드
        vs_manager = VectorStoreManager()

        if not vs_manager.vectorstore_exists():
            print("\n❌ 벡터 저장소가 없습니다.")
            print("먼저 문서를 인덱싱해주세요: python src/main.py --index\n")
            return False

        vs_manager.load_vectorstore()

        # RAG 체인 생성 및 실행
        rag = RAGChain(vs_manager)
        result = rag.query(question)
        rag.print_result(result)

        return True

    except Exception as e:
        print(f"\n❌ 질의응답 실패: {str(e)}\n")
        return False


def interactive_mode():
    """
    대화형 모드 실행
    """
    try:
        # 벡터 저장소 로드
        vs_manager = VectorStoreManager()

        if not vs_manager.vectorstore_exists():
            print("\n❌ 벡터 저장소가 없습니다.")
            print("먼저 문서를 인덱싱해주세요: python src/main.py --index\n")
            return False

        print("\n벡터 저장소 로드 중...")
        vs_manager.load_vectorstore()

        # RAG 체인 생성 및 대화형 모드 실행
        rag = RAGChain(vs_manager)
        rag.interactive_mode()

        return True

    except Exception as e:
        print(f"\n❌ 대화형 모드 실패: {str(e)}\n")
        return False


def show_config():
    """
    현재 설정 정보 출력
    """
    print("\n" + "="*60)
    print("⚙️  RAG System Configuration")
    print("="*60)

    config = get_config_info()
    for key, value in config.items():
        print(f"{key:20s}: {value}")

    print("="*60 + "\n")


def main():
    """
    메인 함수
    """
    parser = argparse.ArgumentParser(
        description="OpenAI RAG System - 문서 기반 질의응답 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 문서 인덱싱
  python src/main.py --index

  # 질문하기
  python src/main.py --query "문서의 주요 내용은 무엇인가요?"

  # 대화형 모드
  python src/main.py --interactive

  # 설정 확인
  python src/main.py --config
        """
    )

    # 인자 정의
    parser.add_argument(
        '--index',
        action='store_true',
        help='문서를 로드하고 벡터 저장소를 생성합니다'
    )

    parser.add_argument(
        '--query',
        type=str,
        metavar='QUESTION',
        help='단일 질문에 대한 답변을 생성합니다'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='대화형 모드로 실행합니다'
    )

    parser.add_argument(
        '--config',
        action='store_true',
        help='현재 설정을 출력합니다'
    )

    # 인자 파싱
    args = parser.parse_args()

    # 인자가 없으면 도움말 출력
    if len(sys.argv) == 1:
        parser.print_help()
        return

    # 설정 출력
    if args.config:
        show_config()
        return

    # 인덱싱
    if args.index:
        success = index_documents()
        sys.exit(0 if success else 1)

    # 단일 질문
    if args.query:
        success = query_single(args.query)
        sys.exit(0 if success else 1)

    # 대화형 모드
    if args.interactive:
        success = interactive_mode()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
