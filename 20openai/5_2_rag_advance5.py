# LLM 최종 답변을 생성하기 전에 필요한 문서를 검색(retrieval)하는 과정
# 그 중에서 질문(query)와 문서(document)를 다루는 기술들이 1~5번 기법이야.
# 즉, **LLM 프롬프트에 들어갈 context를 “더 잘 찾고, 정제하고, 압축하는 것”**이 목표.


# 1. Query Transformation    ----> (질문 변화) - 검색 최적화
# 2. Multi-Query             ----> (다중 질의) - 검색 범위 확대 
# 3. Self-RAG                ---->  (자기 보정) - 문서 관련성 평가
# 4. Contextual Compression  ----> (문맥압축) - 관련 부분만 추출
# 5. Fusion Retriever        ----> (융합 검색) 키워드 + 벡터 검색 결합

########## 5. Fusion Retriever        ----> (융합 검색) 키워드 + 벡터 검색 결합
## 참고 !!!! 
# C:\00AI\LLM\20openai\4_7_advance_fusion_retriever.ipynb 


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
    loader_kwargs={'encoding':'utf-8'}
    # recursive = True # ---> 이거 적어두면 하위 폴더까지 다 보겠다는 의미 
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


# 벡터DB를 retriever로 사용할 수 있도록 만듦?
base_retriever = vectorstore.as_retriever(
    search_type = 'similarity',
    search_kwargs = {'k':3}
)


# LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0) 
print(f'set up complete!!')

# 최종 답변
rag_prompt = ChatPromptTemplate.from_messages([
    ('system','제공된 문맥을 바탕으로 한국어로 답변하세요'),
    ('human', '문맥:\n{context}\n\n질문:{question}\n\n답변:')
])


################################################################
########## 5. Fusion Retriever        ----> (융합 검색) 키워드 + 벡터 검색 결합

# 융합 검색
# 1 개별 검색 결과 비교
# 2 융합 결과로 답변 생성

from langchain_community.retrievers import BM25Retriever
# BM25 리트리버         : 키워드기반
# Vector 리트리버       : 의미기반
bm25_retriever = BM25Retriever.from_documents(doc_splits)
bm25_retriever_k = 3

question = 'VectorDB의 종류를 알려주세요'
 # 백터 검색
vector_docs = base_retriever.invoke(question)
# BM25 검색
bm25_docs = bm25_retriever.invoke(question)
fusion_scores = {}
# 백터 검색 결과 점수
for rank, doc in enumerate(vector_docs):
    doc_key = doc.page_content[:50]
    score = 1 / (60 + rank)
    fusion_scores[doc_key] = fusion_scores.get(doc_key,0) + score
# BM25 검색 결과 점수
for rank, doc in enumerate(bm25_docs):
    doc_key = doc.page_content[:50]
    score = 1 / (60 + rank)
    fusion_scores[doc_key] = fusion_scores.get(doc_key,0) + score
# 점수로 정렬
sorted_docs =  sorted(
    fusion_scores.items(), key=lambda x : x[1], reverse=True
)

# print(f'fusion docs 결과 : {sorted_docs}')
docs = []
for doc, score in sorted_docs[:3]:
    docs.append(doc)

inputs = '\n\n---\n\n'.join(docs)
print(inputs)

rag_prompt_chain = rag_prompt | llm | StrOutputParser()
result = rag_prompt_chain.invoke({'context' : inputs, 'question':question})
print(result)