# LLM 최종 답변을 생성하기 전에 필요한 문서를 검색(retrieval)하는 과정
# 그 중에서 질문(query)와 문서(document)를 다루는 기술들이 1~5번 기법이야.
# 즉, **LLM 프롬프트에 들어갈 context를 “더 잘 찾고, 정제하고, 압축하는 것”**이 목표.


# 1. Query Transformation    ----> (질문 변화) - 검색 최적화 
# 2. Multi-Query             ----> (다중 질의) - 검색 범위 확대
# 3. Self-RAG                ---->  (자기 보정) - 문서 관련성 평가
# 4. Contextual Compression  ----> (문맥압축) - 관련 부분만 추출
# 5. Fusion Retriever        ----> (융합 검색) 키워드 + 벡터 검색 결합


############################ # 3. Self-RAG  ---->  (자기 보정) - 문서 관련성 평가
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
path = "C:/00AI/LLM/20openai/sample_docs_20251128"
loader = DirectoryLoader(
    path=path,
    glob='**/*.txt',
    loader_cls=TextLoader,
    loader_kwargs={'encoding':'utf-8'})
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


# 벡터DB를 retriever로 사용할 수 있도록 만듦?
base_retriever = vectorstore.as_retriever(
    search_type = 'similarity',
    search_kwargs = {'k':3}
)


# LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0) 
print(f'set up complete!!')





################################################################
########## # 3. Self-RAG  ---->  (자기 보정) - 문서 관련성 평가
print (f'==================== 3. self-RAG ================')
print (f'==================== 검색된 문서의 관련성을 평가하여 필터링합니다.==================== \n')


# 프롬프트
check_prompt = ChatPromptTemplate.from_template('''
            다음 문서가 질문에 관련이 있는지 평가하세요
            'yes' 또는 'no'로만 답변하세요.
                                            
            문서: {document}
            질문: {question}
            관련성: ''')

# LCEL 체인 구성
check_prompt_chain = check_prompt | llm | StrOutputParser()



# 관련있는 문서만 필터링 (yes로 한 문서만 쓰겠다)
def filter_relevant_docs(docs, question):
    ''' 관련 있는 문서만 필터링 in retriever'''
    relevant = []
    for doc in docs : 
        result = check_prompt_chain.invoke({'document': doc.page_content, 'question': question})
        is_relevant = 'yes' in result.lower()
        print (f' --> {doc.page_content} : {"Relevant" if is_relevant else "Not Relevant"}')
        if is_relevant:
            relevant.append(doc)
    return relevant


# 관련성을 평가 후 답변 생성

# 1. 문서를 검색 (리트리버를 이용해서)
question = 'rag란 무엇인가?' #'환율이 어떻게 돼?' #
docs = base_retriever.invoke(question)
print(f'retriever가 찾은 문서의 개수 : {len(docs)}개')

# 관련성 평가
relevant_docs = filter_relevant_docs(docs, question)
print (f'relevant_docs 개수 : {len(relevant_docs)}개')

if not relevant_docs:
    raise ValueError(f"[사용자 질문]: {question} \n --> 관련있는 문서가 없어서 답변을 종료합니다. 다른 질문을 입력하세요.")



# 유틸리티 출력 함수
def format_docs(docs):
    '''문서를 문자열로 포맷팅'''
    return '\n\n---\n\n'.join([doc.page_content for doc in docs])
context = format_docs(relevant_docs)

# 답변생성
# RAG프롬프트
rag_prompt = ChatPromptTemplate.from_messages([
    ('system', '제공된 문맥을 바탕으로 한국어로 답변하세요'),
    ('human', '문맥:\n{context}\n\n질문:{question}\n\n답변:')
])

answer_chain = rag_prompt | llm | StrOutputParser()

answer = answer_chain.invoke({'context':context, 'question':question})
print(f'질문: {question}')
print (f'답변: {answer}')
sources=[os.path.basename(d.metadata.get('source', 'unknown')) for d in relevant_docs]
print( f'근거:{sources}')

