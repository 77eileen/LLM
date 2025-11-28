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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20, # 10~20%
    separators=['\n\n', '\n', '.', ',', ' ']
)

# 스플릿 = chunk
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
retriever = vectorstore.as_retriever(
    search_type = 'similarity',
    search_kwargs = {'k':3}
)

# 프롬프트 설정
prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'Answer question based on the given context in Korean'),
    ('human', 'Context: \n{context}\n\nQuestion:{question}\n\nAnswer:')
])

# 유틸리티 출력 함수
def format_docs(docs):
    '''문서를 문자열로 포맷팅'''
    return '\n\n---\n\n'.join([doc.page_content for doc in docs])


# LCEL 방식 Runnable 객체 : 실행 invoke --> 파이프라인
# llm = ChatOpenAI(model='gpt-4o-mini', temperature=0) 이렇게 정의하고 하기 파이프라인에서 llm이라 써도됨
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()} #?????????????
    | prompt_template
    | ChatOpenAI(model='gpt-4o-mini', temperature=0)
    | StrOutputParser()
)

def ask_question(question):
    '''질문에 대한 답변 생성'''
    answer = rag_chain.invoke(question)  # 질문이 들어오면 상기 rag_chain 을 쫘악 거침
    retrieved_docs =retriever.invoke(question) # 출처 확인
    sources = [os.path.basename(doc.metadata.get('source', 'unknown'))for doc in retrieved_docs]
        # os.pathe.basename() 이렇게 감싸주면 fullpath가 아니라 파일명만 가져옴
        # 파일을 저장하면? 자동으로 metadata가 생김?
    return answer, sources
    # print(f'Question: {question}')
    # print(f'Answer: {answer}')




# 각 질문에 대한 답변 생성
test_question = [
    'RAG란 무엇인가요?',
    'LangGraph의 핵심 개념을 설명해주세요',
    '프롬프트 엔지니어링 기법에는 어떤 것들이 있나요?'
]

for i, question in enumerate(test_question, 1):
    print(f'\n\n---> 질문{i}: {question}')
    answer, sources = ask_question(question)
    print(f'---> 답변: {answer}')
    print(f'\n---> 출처: {sources}')


# RAG를 쓰면, 제공한 소스 (문서내에서)에서 답변을 함
# 노트북LLM 이라는 AI 있음. https://notebooklm.google/ --> RAG와 유사