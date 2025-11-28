# LLM 최종 답변을 생성하기 전에 필요한 문서를 검색(retrieval)하는 과정
# 그 중에서 질문(query)와 문서(document)를 다루는 기술들이 1~5번 기법이야.
# 즉, **LLM 프롬프트에 들어갈 context를 “더 잘 찾고, 정제하고, 압축하는 것”**이 목표.

# 1. Query Transformation    ----> (질문 변화) - 검색 최적화 
# 2. Multi-Query             ----> (다중 질의) - 검색 범위 확대 
# 3. Self-RAG                ---->  (자기 보정) - 문서 관련성 평가
# 4. Contextual Compression  ----> (문맥압축) - 관련 부분만 추출
# 5. Fusion Retriever        ----> (융합 검색) 키워드 + 벡터 검색 결합


########### 2. Multi-Query             ----> (다중 질의) - 검색 범위 확대

import os
import warnings
import pickle    # chunk, vectorDB 저장한것 사용
from dotenv import load_dotenv

# 경고메세지 삭제
warnings.filterwarnings('ignore')
load_dotenv()

# openapi key 확인
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError('.env확인,  key없음')

# 필수 라이브러리 로드
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 문서로드
# chunking (텍스트를 분할)
# embedding 및 VectorDB
# retriever
# LLM 설정



# 문서로드
# print(os.path.abspath(__file__))
script_dir= os.path.dirname(os.path.abspath(__file__))  # __file__ ===> .py 파일에서만 가능함. 노트북안됨.
docs_path = os.path.join(script_dir, 'sample_docs_20251128') #(script_dir, 'sample_docs_20251128') 여기에 폴더 경로가 상위, 하위 더 있으면 (script_dir, '상위폴더명', 'sample_docs_20251128', '하위폴더명') 쓰기
print(f'---> docs_path: {docs_path}')

loader = DirectoryLoader(
    docs_path,
    glob='**/*.txt',
    loader_cls=TextLoader,
    loader_kwargs={'encoding':'utf-8'}
)

document = loader.load()
print(f'---> 읽은 문서의 수: {len(document)}')


# 텍스트 분할 chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20, # 10~20%
    separators=['\n\n', '\n', '.', ',', ' ']
)

doc_splits = text_splitter.split_documents(document)
print(f'---> chunkig 개수 : {len(doc_splits)}')


# 임베딩 및 vectorstore (벡터DB)
embedding_model =OpenAIEmbeddings(model='text-embedding-3-small')

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name='basic_rag_collection',
    embedding=embedding_model
)


# 벡터DB를 retriever 설정?
base_retriever = vectorstore.as_retriever(
    search_type = 'similarity',
    search_kwargs = {'k':3}
)


# LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0) 
print(f'set up complete!!')


# 유틸리티 출력 함수
def format_docs(docs):
    '''문서를 문자열로 포맷팅'''
    return '\n\n---\n\n'.join([doc.page_content for doc in docs])

# RAG프롬프트
rag_prompt = ChatPromptTemplate.from_messages([
    ('system', '제공된 문맥을 바탕으로 한국어로 답변하세요'),
    ('human', '문맥:\n{context}\n\n질문:{question}\n\n답변:')
])





################################################################
########## # 2. Multi-Query             ----> (다중 질의) - 검색 범위 확대
# 다중쿼리 생성 프롬프트
multi_query_prompt = ChatPromptTemplate.from_template(
    '''다음 질문에 대해 5가지 다른 관점의 검색 쿼리를 생성하세요.
    각 쿼리는 새 줄(new line)로 구분하여 출력하세요.
    번호나 설명 없이 쿼리만 출력하세요

    원본 질문: {question}
    다른 관점의 쿼리들: 
    ''')

# lag chain 구성 LCEL
multi_query_chain = multi_query_prompt | llm | StrOutputParser()


def multi_query_rag(question):
    '''다중 질의(쿼리) 검색'''
    # 1. 다중 쿼리 생성
    multi_queies_text = multi_query_chain.invoke({'question':question})
    multi_queies = [ q.strip() for q in multi_queies_text.strip().split('\n') if q.strip()]
    print(multi_queies)


    # 각 쿼리(질문)로 검색하고 결과를 통합 (중복제거하는 기능도 추가)
    all_docs =[]
    seen_contents = set()  #중복제거
    for query in multi_queies :
        docs = base_retriever.invoke(query)
        for doc in docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                all_docs.append(doc)

    print(f'검색된 문서의 개수 : {len(all_docs)}')

    # 리트리버 답변 생성 : 추출된 문서의 상위 3개만 사용
    context = format_docs(all_docs[:2])
    print(context)
    answer_chain = rag_prompt | llm | StrOutputParser()
    answer = answer_chain.invoke({'context':context, 'question':question})
    return answer, [os.path.basename(d.metadata.get('source', 'unknown')) for d in all_docs]





#####################################
# 각 질문에 대한 답변 생성
test_question = [
    '프롬프트 엔지니어링 기법에는 어떤 것들이 있나요?',
    'vectorDB란?'
]

for i, question in enumerate(test_question, 1):
    print(f'\n\n---> 질문{i}: {question}')
    answer, sources = multi_query_rag(question)
    print(f'---> 답변: {answer}')
    print(f'\n---> 출처: {sources}')

