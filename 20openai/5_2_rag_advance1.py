# LLM 최종 답변을 생성하기 전에 필요한 문서를 검색(retrieval)하는 과정
# 그 중에서 질문(query)와 문서(document)를 다루는 기술들이 1~5번 기법이야.
# 즉, **LLM 프롬프트에 들어갈 context를 “더 잘 찾고, 정제하고, 압축하는 것”**이 목표.


# 1. Query Transformation    ----> (질문 변화) - 검색 최적화 
# 2. Multi-Query             ----> (다중 질의) - 검색 범위 확대
# 3. Self-RAG                ---->  (자기 보정) - 문서 관련성 평가
# 4. Contextual Compression  ----> (문맥압축) - 관련 부분만 추출
# 5. Fusion Retriever        ----> (융합 검색) 키워드 + 벡터 검색 결합


########## 1. Query Transformation    ----> (질문 변화) - 검색 최적화

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
                            # page_content : 문서의 실제 텍스트 내용(본문)
                            # metadata : 파일 이름, 경로, 페이지 번호 등 부가정보



# 질문 재작성 프롬프트
rewrite_prompt = ChatPromptTemplate.from_template('''
다음 질문을 검색에 더 적합한 형태로 변환해 주세요.
키워드 중심으로 명확하게 바꿔주세요.
변환된 검색어만 출력하세요.
- 원본 질문 : {question}
- 변환된 검색어 :                                  
''')


# ?????????????????? 앞에 basic이랑 비교해보기.. 앞에 basic 왜그렇게 하는지?
rewrite_chain = rewrite_prompt | llm | StrOutputParser()


# RAG프롬프트
rag_prompt = ChatPromptTemplate.from_messages([
    ('system', '제공된 문맥을 바탕으로 한국어로 답변하세요'),
    ('human', '문맥:\n{context}\n\n질문:{question}\n\n답변:')
])



################################################################
########## 1. Query Transformation    ----> (질문 변화) - 검색 최적화

def query_transform(question):
    '''Query Transformation  (질문 변화) - 검색 최적화'''    
    print ('1. Query Transformation')
    print ('사용자 질문을 검색에 최적화된 형태로 변환합니다.\n')

    # 1) 질문 변환
    # 틀렸음 docs = rewrite_chain.invoke({'question':question})
    # 틀렸음 context = format_docs(docs)
    transformed = rewrite_chain.invoke({'question':question})
    print(f'원본 질문: {question}')
    print(f'변환된 질문: {transformed}')

    # 2) 변환된 질문으로 리트리버에 검색 후 rag_prompt 한것을 llm으로
    docs = base_retriever.invoke(transformed)
    context = format_docs(docs)
    answer_chain = rag_prompt | llm | StrOutputParser()

    answer = answer_chain.invoke({'context':context, 'question':question})
    return answer, [os.path.basename(d.metadata.get('source', 'unknown')) for d in docs]

    ######## 5_1_basic 과 비교 / 차이점 알기!!!!!!!!!!!!
    # rag_chain = (
    #     {'context': retriever | format_docs, 'question': RunnablePassthrough()} #?????????????
    #     | prompt_template
    #     | ChatOpenAI(model='gpt-4o-mini', temperature=0)
    #     | StrOutputParser()
    # )



#####################################
# 각 질문에 대한 답변 생성
test_question = [
    'RAG 어떻게 쓰나요?',
    'LangGraph 뭐하는거야?',
    '프롬프트 엔지니어링 기법에는 어떤 것들이 있나요?'
]

for i, question in enumerate(test_question, 1):
    print(f'\n\n---> 질문{i}: {question}')
    answer, sources = query_transform(question)
    print(f'---> 답변: {answer}')
    print(f'\n---> 출처: {sources}')

