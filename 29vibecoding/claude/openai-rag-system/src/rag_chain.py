"""
RAG Chain for Question Answering
문서 기반 질의응답 체인 모듈
"""

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from vectorstore import VectorStoreManager
from config import LLM_MODEL, OPENAI_API_KEY, TOP_K_RESULTS


class RAGChain:
    """RAG 질의응답 체인 클래스"""

    def __init__(
        self,
        vectorstore_manager: VectorStoreManager,
        model: str = LLM_MODEL,
        temperature: float = 0.0
    ):
        """
        RAG 체인 초기화

        Args:
            vectorstore_manager: 벡터 저장소 매니저
            model: LLM 모델 이름
            temperature: 생성 온도 (0.0 = 결정적, 1.0 = 창의적)
        """
        self.vectorstore_manager = vectorstore_manager
        self.model = model

        # LLM 초기화
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY
        )

        # Retriever 설정
        self.retriever = self.vectorstore_manager.get_retriever(k=TOP_K_RESULTS)

        # Prompt 템플릿 설정
        self.prompt = self._create_prompt_template()

        # RAG 체인 생성
        self.chain = self._create_chain()

        print(f"🤖 RAG 체인 초기화 완료 (모델: {model})")

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        RAG 프롬프트 템플릿 생성

        Returns:
            ChatPromptTemplate
        """
        template = """당신은 문서를 기반으로 질문에 답변하는 AI 어시스턴트입니다.

다음 문서 조각들을 참고하여 질문에 답변해주세요.
답변은 문서에 있는 정보만을 사용하고, 없는 정보는 "문서에 해당 정보가 없습니다"라고 답변하세요.
답변은 친절하고 명확하게 한국어로 작성해주세요.

문서 내용:
{context}

질문: {question}

답변:"""

        return ChatPromptTemplate.from_template(template)

    def _format_docs(self, docs: List[Document]) -> str:
        """
        검색된 문서들을 포맷팅

        Args:
            docs: Document 리스트

        Returns:
            포맷팅된 문서 문자열
        """
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            formatted.append(f"[문서 {i} - {source}]\n{content}")

        return "\n\n".join(formatted)

    def _create_chain(self):
        """
        RAG 체인 생성 (LCEL 방식)

        Returns:
            Runnable 체인
        """
        # LCEL (LangChain Expression Language) 사용
        chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def query(self, question: str) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성

        Args:
            question: 사용자 질문

        Returns:
            답변 및 소스 문서 정보
        """
        try:
            print(f"\n💭 질문: {question}")

            # 관련 문서 검색
            retrieved_docs = self.retriever.invoke(question)

            print(f"📚 검색된 문서: {len(retrieved_docs)}개")

            # RAG 체인 실행
            answer = self.chain.invoke(question)

            # 소스 문서 정보 추출
            sources = self._extract_sources(retrieved_docs)

            result = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "retrieved_docs": retrieved_docs
            }

            return result

        except Exception as e:
            print(f"❌ 질의응답 실패: {str(e)}")
            raise

    def _extract_sources(self, docs: List[Document]) -> List[Dict[str, str]]:
        """
        문서에서 소스 정보 추출

        Args:
            docs: Document 리스트

        Returns:
            소스 정보 딕셔너리 리스트
        """
        sources = []
        seen_sources = set()

        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')

            # 중복 제거
            if source not in seen_sources:
                sources.append({
                    "source": source,
                    "file_type": doc.metadata.get('file_type', 'unknown'),
                    "preview": doc.page_content[:100] + "..."
                })
                seen_sources.add(source)

        return sources

    def print_result(self, result: Dict[str, Any]):
        """
        결과를 보기 좋게 출력

        Args:
            result: query() 결과 딕셔너리
        """
        print(f"\n{'='*60}")
        print(f"❓ 질문: {result['question']}")
        print(f"\n{'='*60}")
        print(f"💡 답변:\n{result['answer']}")
        print(f"\n{'='*60}")
        print(f"📖 참고 문서:")

        for i, source in enumerate(result['sources'], 1):
            print(f"\n  [{i}] {source['source']}")
            print(f"      미리보기: {source['preview']}")

        print(f"{'='*60}\n")

    def interactive_mode(self):
        """
        대화형 모드 실행
        """
        print("\n" + "="*60)
        print("🤖 RAG 대화형 모드")
        print("="*60)
        print("질문을 입력하세요. 종료하려면 'exit', 'quit', 'q'를 입력하세요.\n")

        while True:
            try:
                question = input("질문> ").strip()

                if not question:
                    continue

                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 대화를 종료합니다.")
                    break

                # 질의응답 실행
                result = self.query(question)
                self.print_result(result)

            except KeyboardInterrupt:
                print("\n\n👋 대화를 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {str(e)}\n")


def test_rag_chain():
    """RAG 체인 테스트"""
    print("=== RAG Chain 테스트 ===\n")

    from document_loader import DocumentLoader

    try:
        # 1. 벡터 저장소 로드 또는 생성
        vs_manager = VectorStoreManager()

        if vs_manager.vectorstore_exists():
            print("기존 벡터 저장소 로드...")
            vs_manager.load_vectorstore()
        else:
            print("새로운 벡터 저장소 생성...")
            loader = DocumentLoader()
            chunks = loader.load_and_split()
            vs_manager.create_vectorstore(chunks)
            vs_manager.save_vectorstore()

        # 2. RAG 체인 생성
        print("\nRAG 체인 생성...")
        rag = RAGChain(vs_manager)

        # 3. 테스트 질문
        test_question = "문서의 주요 내용은 무엇인가요?"
        print(f"\n테스트 질문: {test_question}")

        result = rag.query(test_question)
        rag.print_result(result)

        print("\n✅ RAG 체인 테스트 성공!")

    except FileNotFoundError as e:
        print(f"⚠️  {str(e)}")
        print("data/ 폴더에 문서를 추가한 후 다시 시도하세요.")
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")


if __name__ == "__main__":
    test_rag_chain()
