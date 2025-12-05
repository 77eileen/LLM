# 📚 HuggingFace DailyPapers 데이터 수집 모듈 (01_data_collection)

## 목차
1. [개요](#1-개요)
2. [데이터 소스 및 구조](#2-데이터-소스-및-구조)
3. [구현 상세 계획](#3-구현-상세-계획)
4. [데이터 검증 및 에러 처리](#4-데이터-검증-및-에러-처리)
5. [로깅 전략](#5-로깅-전략)
6. [테스트 및 검증](#6-테스트-및-검증)

---

## 1. 개요

### 1.1. 목적
- **HuggingFace Daily Papers**에서 최신 Weekly 논문 정보를 자동으로 수집
- 수집된 데이터를 **Vector DB (ChromaDB)** 인덱싱에 최적화된 형식으로 저장
- 주간 단위로 구조화된 데이터 관리 체계 구축

### 1.2. 주요 기능
- Weekly Papers 목록 URL 자동 생성
- 각 논문의 상세 정보 크롤링 (제목, Abstract, GitHub URL, Upvote 등)
- 논문 Abstract 기반 **자동 키워드(태그) 추출**
- 구조화된 텍스트 파일 및 메타데이터 CSV 생성

### 1.3. 기술 스택
| 분류 | 라이브러리 | 용도 |
|------|-----------|------|
| **HTTP 요청** | `requests` | HuggingFace 페이지 요청 |
| **HTML 파싱** | `BeautifulSoup4` | HTML 구조 파싱 및 데이터 추출 |
| **키워드 추출** | `TF-IDF` / `KeyBERT` | Abstract 기반 중요 키워드 자동 추출 |
| **데이터 저장** | `pandas` | CSV 메타데이터 관리 |
| **로깅** | `logging` | 크롤링 과정 로그 기록 |

---

## 2. 데이터 소스 및 구조

### 2.1. HuggingFace DailyPapers Weekly URL
- **기본 URL 패턴:** `https://huggingface.co/papers/week/{YYYY-Www}`
  - 예시: `https://huggingface.co/papers/week/2025-W49`
  - `YYYY`: 연도 (4자리)
  - `ww`: 주차 (01-52, zero-padding 필수)

### 2.2. 데이터 저장 형식 비교 및 선택

#### 2.2.1. TXT vs JSON 형식 비교

| 비교 기준 | TXT (Hybrid) | JSON | 선택 |
|----------|-------------|------|------|
| **파싱 용이성** | 수동 split/parse 필요 | `json.load()` 한 줄 | ✅ **JSON** |
| **데이터 구조화** | 혼합 형식 (Context: plain text + MetaData: JSON) | 완전한 구조화 | ✅ **JSON** |
| **ChromaDB 호환** | 추가 파싱 로직 필요 | `Document` 객체로 직접 변환 | ✅ **JSON** |
| **확장성** | 필드 추가 시 파싱 로직 수정 | 스키마만 확장하면 됨 | ✅ **JSON** |
| **에러 처리** | 수동 검증 | JSON Schema, Pydantic 활용 가능 | ✅ **JSON** |
| **타입 안정성** | 문자열 → 수동 변환 | 자동 타입 처리 (int, str, list 등) | ✅ **JSON** |
| **범용성** | Python 외 파싱 어려움 | 모든 프로그래밍 언어 지원 | ✅ **JSON** |
| **가독성** | 읽기 편함 | 약간 덜 직관적 | TXT |
| **파일 크기** | 약간 작음 | 약간 큼 (~5-10% 증가) | TXT |

**결론:** JSON 방식을 채택합니다.
- RAG 파이프라인에서 ChromaDB Document 생성 시 직접 매핑 가능
- 데이터 검증 및 확장성에서 압도적 우위
- 파일 크기 증가는 무시할 수 있는 수준 (논문당 ~100-200 bytes 차이)

#### 2.2.2. 최종 저장 디렉토리 구조
```
01_data/
└── documents/
    └── {YYYY-Www}/              # 예: 2025-W49/
        ├── doc2549001.json      # 논문 문서 (JSON 형식)
        ├── doc2549002.json
        └── ...
```

### 2.3. JSON 문서 파일 구조 (*.json)

#### 2.3.1. 파일 명명 규칙
- **형식:** `doc{YY}{ww}{NNN}.json`
  - `YY`: 연도 마지막 2자리 (예: 2025 → 25)
  - `ww`: 주차 2자리 (예: W49 → 49)
  - `NNN`: 일련번호 3자리 zero-padding (예: 001, 002, ..., 999)
- **예시:**
  - 2025년 49주차 첫 번째 논문: `doc2549001.json`
  - 2025년 49주차 다섯 번째 논문: `doc2549005.json`

#### 2.3.2. JSON 스키마 (Schema)
```json
{
  "context": "논문의 Abstract 전문 텍스트...",
  "metadata": {
    "paper_name": "논문 제목",
    "github_url": "https://github.com/repo/name",
    "huggingface_url": "https://huggingface.co/papers/2511.18538",
    "upvote": 123,
    "tags": ["keyword1", "keyword2", "keyword3"]
  }
}
```

#### 2.3.3. 필드 상세 설명

| 필드명 | 타입 | 필수 | 설명 |
|--------|------|------|------|
| `context` | `str` | ✅ | 논문 Abstract 전문 (RAG 검색 대상) |
| **metadata** | `object` | ✅ | 논문 메타데이터 객체 |
| `metadata.pape_rname` | `str` | ✅ | 논문 제목 |
| `metadata.github_url` | `str` | ❌ | GitHub 레포 URL (없으면 빈 문자열) |
| `metadata.huggingface_url` | `str` | ✅ | HuggingFace 논문 페이지 URL |
| `metadata.upvote` | `int` | ✅ | 추천 수 (0 이상) |
| `metadata.tags` | `array[str]` | ✅ | 키워드 리스트 (정확히 3개) |

---

## 3. 구현 상세 계획

### 3.1. 전체 크롤링 파이프라인

```
[1] 주간 URL 생성
    ↓
[2] Weekly Papers 목록 페이지 파싱
    ↓
[3] 각 논문 URL 리스트 추출
    ↓
[4] 개별 논문 상세 페이지 파싱 (병렬 처리 가능)
    ↓
[5] Abstract 기반 키워드 추출
    ↓
[6] JSON 파일 저장
```

### 3.2. Step 1: 주간 URL 생성 및 Papers 목록 수집

#### 3.2.1. Weekly URL 생성 로직
```python
from datetime import datetime

def get_current_week_url():
    """현재 주차의 HuggingFace Weekly Papers URL 생성"""
    now = datetime.now()
    year = now.year
    week = now.isocalendar()[1]  # ISO week number
    return f"https://huggingface.co/papers/week/{year}-W{week:02d}"
```

#### 3.2.2. Papers 목록 추출 (BeautifulSoup)
- **타겟 CSS Selector:** `a.line-clamp-3`
- **추출 정보:**
  - `href` 속성: 논문 상세 페이지 경로 (예: `/papers/2511.18538`)
  - `text`: 논문 제목

**구현 예시:**
```python
from bs4 import BeautifulSoup
import requests

def fetch_paper_urls(weekly_url):
    """Weekly 페이지에서 모든 논문 URL 추출"""
    response = requests.get(weekly_url, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')

    paper_links = []
    for link in soup.select('a.line-clamp-3'):
        href = link.get('href')
        title = link.get_text(strip=True)
        full_url = f"https://huggingface.co{href}"
        paper_links.append({"title": title, "url": full_url})

    return paper_links
```

### 3.3. Step 2: 개별 논문 상세 정보 추출

#### 3.3.1. 타겟 데이터 및 CSS Selector

| 추출 대상 | CSS Selector (또는 XPath) | 비고 |
|-----------|--------------------------|------|
| **Abstract (Context)** | `div p` (section 하위) | 여러 `<p>` 태그를 모두 결합 |
| **GitHub URL** | `a[href*="github.com"]` | GitHub 링크만 필터링 (없을 수 있음) |
| **Upvote** | `div.upvote` 또는 특정 클래스 | 정수 변환 필요 |
| **HuggingFace URL** | 현재 페이지 URL | 파라미터로 전달 |

**기존 XPath 참고:**
- Context: `/html/body/div/main/div/section[1]/div/div[2]/div/p`
- GitHub URL: `/html/body/div/main/div/section[1]/div/div[3]/a[4]`
- Upvote: `/html/body/div/main/div/section[1]/div/div[1]/div[3]/div/a/div/div`

#### 3.3.2. 구현 예시
```python
def fetch_paper_details(paper_url):
    """개별 논문 상세 페이지에서 정보 추출"""
    response = requests.get(paper_url, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Abstract 추출 (여러 <p> 태그를 하나로 결합)
    abstract_section = soup.select_one('section div')
    abstract = ' '.join([p.get_text(strip=True) for p in abstract_section.find_all('p')])

    # GitHub URL 추출 (선택적)
    github_link = soup.select_one('a[href*="github.com"]')
    github_url = github_link['href'] if github_link else ""

    # Upvote 추출
    upvote_elem = soup.select_one('div.upvote')  # 실제 클래스명 확인 필요
    upvote = int(upvote_elem.get_text(strip=True)) if upvote_elem else 0

    return {
        "abstract": abstract,
        "github_url": github_url,
        "huggingface_url": paper_url,
        "upvote": upvote
    }
```

### 3.4. Step 3: 키워드 추출 (tag1, tag2, tag3)

#### 3.4.1. TF-IDF 기반 키워드 추출
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_top_keywords(text, top_n=3):
    """TF-IDF 기반 상위 N개 키워드 추출"""
    # 단어 단위로 분리
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])

    # 상위 N개 단어 추출
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    top_indices = np.argsort(scores)[-top_n:][::-1]

    keywords = [feature_names[i] for i in top_indices]
    return keywords
```

#### 3.4.2. (선택) KeyBERT 기반 고급 키워드 추출
```python
from keybert import KeyBERT

def extract_keywords_keybert(text, top_n=3):
    """KeyBERT를 이용한 키워드 추출 (더 정확함)"""
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2),
                                         stop_words='english', top_n=top_n)
    return [kw[0] for kw in keywords]
```

### 3.5. Step 4: 파일 저장 (JSON 방식)

#### 3.5.1. JSON 파일 저장
```python
import os
import json
from datetime import datetime

def save_document_json(paper_data, week_str, index, crawler_version="1.0"):
    """논문 데이터를 JSON 파일로 저장"""
    # 파일명 생성: doc{YY}{ww}{NNN}.json
    year = week_str[:4]
    week = week_str.split('-W')[1]
    doc_id = f"doc{year[2:]}{week}{index+1:03d}"
    doc_filename = f"{doc_id}.json"

    # 디렉토리 생성
    save_dir = f"01_data/documents/{week_str}"
    os.makedirs(save_dir, exist_ok=True)

    # JSON 구조 생성
    document = {
        "context": paper_data['abstract'],
        "metadata": {
            "papername": paper_data['title'],
            "github_url": paper_data.get('github_url', ""),
            "huggingface_url": paper_data['huggingface_url'],
            "upvote": paper_data['upvote'],
            "tags": paper_data['tags']  # 리스트 형태로 저장
        }
    }

    # JSON 파일 저장
    file_path = os.path.join(save_dir, doc_filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON 저장: {doc_filename}")
    return doc_id, doc_filename
```

#### 3.5.2. JSON 파일 읽기 (검증 및 로드)
```python
def load_document_json(file_path):
    """JSON 파일 로드 및 구조 검증"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 필수 필드 검증
    required_fields = ['doc_id', 'context', 'metadata', 'collection_info']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"필수 필드 누락: {field}")

    return data
```

---

## 4. 데이터 검증 및 에러 처리

### 4.1. JSON 스키마 검증 (Pydantic 사용)

**권장 방법:** Pydantic을 사용한 타입 안정성 보장

```python
from pydantic import BaseModel, Field, validator
from typing import List
from datetime import datetime

class Metadata(BaseModel):
    papername: str = Field(..., min_length=1)
    github_url: str = ""
    huggingface_url: str = Field(..., regex=r'^https://huggingface\.co/papers/')
    upvote: int = Field(..., ge=0)
    tags: List[str] = Field(..., min_items=3, max_items=3)

class CollectionInfo(BaseModel):
    week: str = Field(..., regex=r'^\d{4}-W\d{2}$')
    collected_at: str
    crawler_version: str

class PaperDocument(BaseModel):
    doc_id: str = Field(..., regex=r'^doc\d{6}$')
    context: str = Field(..., min_length=100)
    metadata: Metadata
    collection_info: CollectionInfo

    @validator('context')
    def context_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Context cannot be empty')
        return v

# 사용 예시
def validate_and_save(paper_data, week_str, index):
    """Pydantic으로 검증 후 저장"""
    try:
        # 검증
        validated_doc = PaperDocument(**paper_data)

        # 검증 통과 시 저장
        doc_id, filename = save_document_json(
            paper_data=validated_doc.dict(),
            week_str=week_str,
            index=index
        )
        return doc_id, filename

    except ValidationError as e:
        print(f"❌ 데이터 검증 실패: {e}")
        raise
```

### 4.2. 간단한 수동 검증 (Pydantic 미사용 시)
```python
def validate_paper_data(paper_data):
    """논문 데이터 유효성 검사 (수동 방식)"""
    # 1. 필수 필드 존재 확인
    required_fields = ['title', 'abstract', 'huggingface_url', 'upvote', 'tags']
    for field in required_fields:
        if field not in paper_data or paper_data[field] is None:
            raise ValueError(f"필수 필드 누락: {field}")

    # 2. Abstract 최소 길이 검증
    if len(paper_data['abstract'].strip()) < 100:
        raise ValueError(f"Abstract가 너무 짧습니다 (현재: {len(paper_data['abstract'])}자, 최소: 100자)")

    # 3. 태그 개수 및 타입 검증
    if not isinstance(paper_data['tags'], list):
        raise TypeError("tags는 리스트 타입이어야 합니다")

    if len(paper_data['tags']) != 3:
        raise ValueError(f"태그는 정확히 3개여야 합니다 (현재: {len(paper_data['tags'])}개)")

    # 4. Upvote 범위 검증
    if not isinstance(paper_data['upvote'], int) or paper_data['upvote'] < 0:
        raise ValueError(f"Upvote는 0 이상의 정수여야 합니다 (현재: {paper_data['upvote']})")

    # 5. URL 형식 검증
    if not paper_data['huggingface_url'].startswith('https://huggingface.co/papers/'):
        raise ValueError(f"잘못된 HuggingFace URL: {paper_data['huggingface_url']}")

    return True
```

### 4.3. HTTP 요청 재시도 로직
```python
import time

def fetch_with_retry(url, max_retries=3, backoff=2):
    """재시도 로직이 포함된 HTTP 요청"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = backoff ** attempt
                print(f"⚠️ 요청 실패 ({attempt+1}/{max_retries}), {wait_time}초 후 재시도...")
                time.sleep(wait_time)
            else:
                raise Exception(f"최대 재시도 횟수 초과: {url}") from e
```

### 4.4. 에러 처리 전략
- **네트워크 에러:** 최대 3회 재시도 (exponential backoff)
- **파싱 에러:** 해당 논문 스킵 후 로그 기록
- **저장 실패:** 예외 발생 시 롤백 (이미 저장된 파일 삭제 고려)
- **JSON 검증 실패:** Pydantic ValidationError 로그 기록 및 스킵

---

## 5. 로깅 전략

### 5.1. 로그 파일 구조
```
01_data/
└── logs/
    └── crawling_{YYYY-Www}_{timestamp}.log
```

### 5.2. 로깅 설정 예시
```python
import logging
from datetime import datetime

def setup_logging(week_str):
    """크롤링 로그 설정"""
    log_dir = "01_data/logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/crawling_{week_str}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info(f"🚀 크롤링 시작: {week_str}")
```

### 5.3. 주요 로깅 포인트
- 크롤링 시작/종료 시간
- 각 논문 처리 성공/실패
- HTTP 요청 재시도 로그
- 최종 통계 (총 논문 수, 성공/실패 개수)

---

## 6. 테스트 및 검증

### 6.1. 단위 테스트 체크리스트
- [ ] Weekly URL 생성 로직 검증
- [ ] BeautifulSoup 파싱 결과 검증 (샘플 HTML 사용)
- [ ] 키워드 추출 결과 검증 (샘플 Abstract 사용)
- [ ] 파일명 생성 로직 검증 (edge case: 999번째 논문)
- [ ] CSV 저장/로드 검증

### 6.2. 통합 테스트
- [ ] 전체 크롤링 파이프라인 실행 (최소 1개 주차)
- [ ] 생성된 파일 구조 검증
- [ ] ChromaDB 로드 테스트 (다음 단계에서)

### 6.3. 예외 상황 테스트
- [ ] 네트워크 연결 끊김 시나리오
- [ ] GitHub URL이 없는 논문 처리
- [ ] 비정상적으로 짧은 Abstract 처리
- [ ] 중복 크롤링 방지 로직 (이미 존재하는 주차 재실행 시)

---

## 7. ChromaDB 통합 (다음 단계 미리보기)

### 7.1. JSON → ChromaDB Document 변환

JSON 파일을 ChromaDB Document 객체로 직접 변환하는 방법:

```python
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import json
import os

def load_json_to_chromadb(week_str, collection_name="huggingface_papers"):
    """JSON 파일들을 ChromaDB에 로드"""

    # 디렉토리 경로
    docs_dir = f"01_data/documents/{week_str}"

    # Document 리스트 생성
    documents = []

    for filename in os.listdir(docs_dir):
        if not filename.endswith('.json'):
            continue

        file_path = os.path.join(docs_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Document 객체 생성
        doc = Document(
            page_content=data['context'],  # Abstract 텍스트
            metadata={
                'doc_id': data['doc_id'],
                'papername': data['metadata']['papername'],
                'github_url': data['metadata']['github_url'],
                'huggingface_url': data['metadata']['huggingface_url'],
                'upvote': data['metadata']['upvote'],
                'tags': data['metadata']['tags'],  # 리스트 형태 유지
                'week': data['collection_info']['week'],
                'collected_at': data['collection_info']['collected_at']
            }
        )
        documents.append(doc)

    # ChromaDB에 저장
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )

    print(f"✅ ChromaDB 저장 완료: {len(documents)}개 문서")
    return vectorstore
```

### 7.2. JSON 방식의 장점 (ChromaDB 연동 시)

| 항목 | 설명 |
|------|------|
| **직접 매핑** | `json.load()` → `Document` 객체 생성 (파싱 불필요) |
| **메타데이터 확장** | tags 배열을 그대로 metadata에 저장 가능 |
| **타입 안정성** | upvote는 int, tags는 list로 자동 변환 |
| **쿼리 필터링** | ChromaDB 쿼리 시 `{"upvote": {"$gt": 100}}` 같은 조건 사용 가능 |

### 7.3. 검색 예시 (tags 활용)
```python
# 특정 태그를 포함한 논문 검색
def search_by_tag(vectorstore, tag_keyword, top_k=5):
    """태그 키워드로 논문 검색"""
    results = vectorstore.similarity_search(
        query=tag_keyword,
        k=top_k,
        filter={"tags": {"$in": [tag_keyword]}}  # tags 배열에 키워드 포함 여부
    )
    return results

# 사용 예시
results = search_by_tag(vectorstore, "transformer")
for doc in results:
    print(f"- {doc.metadata['papername']} (Upvote: {doc.metadata['upvote']})")
```

---

## 8. 다음 단계 연계

이 모듈에서 생성된 JSON 데이터는 다음 단계에서 활용됩니다:

1. **02_vectordb_indexing:** JSON 파일을 ChromaDB에 임베딩 및 인덱싱
2. **03_rag_pipeline:** 사용자 질문에 대한 관련 논문 검색 (tags 기반 필터링 포함)
3. **04_streamlit_ui:** 트렌드 키워드(tags) 디스플레이 및 클릭 시 관련 논문 조회

**JSON 데이터 흐름:**
```
JSON Files → ChromaDB Documents → Vector Search → RAG Response → Streamlit UI
```

---

## 9. 참고사항

### 9.1. HuggingFace 페이지 구조 변경 대응
- 주기적으로 CSS Selector 유효성 확인 필요
- 파싱 실패 시 로그 확인 후 Selector 업데이트

### 9.2. Rate Limiting
- HuggingFace API/웹사이트의 요청 제한 정책 확인
- 필요 시 요청 간 딜레이 추가 (`time.sleep(1)` 등)

### 9.3. 데이터 갱신 주기
- 초기 프로토타입: 수동 실행
- 향후: GitHub Actions 또는 Cron Job으로 주간 자동 실행

### 9.4. JSON vs TXT 선택 이유 요약
✅ **JSON 방식을 채택한 최종 이유:**
1. ChromaDB Document 객체로 직접 변환 가능 (파싱 로직 불필요)
2. Pydantic 기반 데이터 검증으로 타입 안정성 보장
3. tags를 확장 가능한 배열로 관리 (tag1/tag2/tag3 분리 불필요)
4. 향후 필드 추가 시 스키마만 확장하면 됨 (파싱 로직 수정 불필요)
5. 파일 크기 증가는 논문당 ~100-200 bytes로 무시할 수 있는 수준
