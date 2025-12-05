import os, json
from langchain_core.documents import Document



# 1. 문서 로더

# 프로젝트 루트 (src 상위 폴더)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_FOLDER = os.path.join(PROJECT_ROOT, '01_data', 'documents', '2025')


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_FOLDER = os.path.join(PROJECT_ROOT, '2025-W45')

print(DATA_FOLDER)
print(os.path.exists(DATA_FOLDER))

documents = []
for root, _, files in os.walk(DATA_FOLDER):
    for f in files:
        if f.endswith(".json"):
            data = json.load(open(os.path.join(root, f), encoding="utf-8"))
            documents.append(Document(page_content=data.get("context", "")))


print(f"총 {len(documents)}개의 Document 로드 완료!")
print(documents[:100])

# 원본 전체 문서 내용을 하나로 합침
full_text = "\n".join([doc.page_content for doc in documents])

# 2. chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=10,
    separators=['\n', '\n\n', '.', ',', ' '],
    length_function =lambda x : len(x.split()) # 단어수 기준
)

chunks = splitter.split_text(full_text)
print(f'원본 문서길이 : {len(full_text)}자')
print(f'RecursiveCharacterTextSplitter 결과 : {len(chunks)}개 chunk')


# 청킹 결과 저장
import pickle
output_path = 'chunks_output.pkl'
with open(output_path, 'wb') as f: 
    pickle.dump(chunks, f)
print(f'저장완료 / 파일명 : {output_path}')



# 3. 임베딩



# 4. vectorDB