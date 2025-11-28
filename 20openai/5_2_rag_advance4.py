# LLM 최종 답변을 생성하기 전에 필요한 문서를 검색(retrieval)하는 과정
# 그 중에서 질문(query)와 문서(document)를 다루는 기술들이 1~5번 기법이야.
# 즉, **LLM 프롬프트에 들어갈 context를 “더 잘 찾고, 정제하고, 압축하는 것”**이 목표.


# 1. Query Transformation    ----> (질문 변화) - 검색 최적화
# 2. Multi-Query             ----> (다중 질의) - 검색 범위 확대 
# 3. Self-RAG                ---->  (자기 보정) - 문서 관련성 평가
# 4. Contextual Compression  ----> (문맥압축) - 관련 부분만 추출
# 5. Fusion Retriever        ----> (융합 검색) 키워드 + 벡터 검색 결합

########## # 4. Contextual Compression  ----> (문맥압축) - 관련 부분만 추출

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



################################################################
########## # 4. Contextual Compression  ----> (문맥압축) - 관련 부분만 추출
print (f'==================== 4. Contextual Compression  ================')
print (f'==================== 검색된 문서를 압축합니다 ==================== \n')



########################### 강사님############################## 

# 문맥압축 프롬프트


question = 'VectorDB의 종류를 알려주세요'

# 1. 문맥압축 프롬프트를 실행
compress_prompt = ChatPromptTemplate.from_template(
'''
다음 문서에서 질문과 관련된 부분만 추출하세요.
관련 없는 부분은 제외하고, 관련 있는 내용만 그대로 출력하세요.
관련 내용이 없으면 "관련 없음"이라고 출력하세요.

문서: {document}
질문: {question}

관련 내용:
'''
)

docs = base_retriever.invoke(question)
compressed = []
sources = []
for doc in docs:
    document = doc.page_content
    compress_chain = compress_prompt | llm | StrOutputParser()
    compress_result = compress_chain.invoke({'question' :question, 'document': document })

    if "관련 없음" not in compress_result:
        compressed.append(compress_result) 
        sources.append( os.path.basename(doc.metadata.get('source',"") ))

context = '\n\n---\n\n'.join(compressed)    

# 최종 답변
rag_prompt = ChatPromptTemplate.from_messages([
    ('system','제공된 문맥을 바탕으로 한국어로 답변하세요'),
    ('human', '문맥:\n{context}\n\n질문:{question}\n\n답변:')
])
rag_prompt_chain = rag_prompt | llm | StrOutputParser()
result = rag_prompt_chain.invoke({'context' : context, 'question':question})
print(result, sources)







############################ 내가 작성한 것 ##############################

# 관련있는 문서만
def compact_document(docs, question):
    ''' 관련 있는 문서만 필터링 in retriever'''
    compact = []
    sources = []
    for doc in docs : 
        result = compact_prompt_chain.invoke({'document': doc.page_content, 'question': question})
        is_not_relevant = '관련 없음' in result.lower()
        print (f' --> {result} : {"Not Relevant"  if is_not_relevant else "Relevant"}')
        if not is_not_relevant:
            compact.append(doc)
            sources.append(os.path.basename(doc.metedata.get('source', 'unknown')))
    return compact



# 1. 문서를 검색 (리트리버를 이용해서)
question = 'retriever에 대해서 설명해줘' #'rag란 무엇인가?' #'환율이 어떻게 돼?' 
docs = base_retriever.invoke(question)
print(f'retriever가 찾은 문서의 개수 : {len(docs)}개')

# 관련성 평가
compact_docs = compact_document(docs, question)
print (f'relevant_docs 개수 : {len(compact_docs)}개')

if not compact_docs:
    raise ValueError(f"[사용자 질문]: {question} \n --> 관련있는 문서가 없어서 답변을 종료합니다. 다른 질문을 입력하세요.")



# 유틸리티 출력 함수
def format_docs(docs):
    '''문서를 문자열로 포맷팅'''
    return '\n\n---\n\n'.join([doc.page_content for doc in docs])
context = format_docs(compact_docs)



# 최종 답변생성
# RAG프롬프트
rag_prompt = ChatPromptTemplate.from_messages([
    ('system', '제공된 문맥을 바탕으로 한국어로 답변하세요'),
    ('human', '문맥:\n{context}\n\n질문:{question}\n\n답변:')
])

answer_chain = rag_prompt | llm | StrOutputParser()

answer = answer_chain.invoke({'context':context, 'question':question})
print(f'질문: {question}')
print (f'답변: {answer}')
sources=[os.path.basename(d.metadata.get('source', 'unknown')) for d in compact_docs]
print( f'근거:{sources}')

