# ==========================================
# HuggingFace DailyPapers Weekly 크롤링 스크립트
# (with NLTK + TF-IDF 고급 키워드 추출 버전)
# ==========================================

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# ---------------------------
# 🔥 추가된 라이브러리 (TF-IDF + NLTK)
# ---------------------------
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK 다운로드
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

TAG_COUNT = 3


# ==========================================
# 1) 로그 설정
# ==========================================
def setup_logging(week_str: str) -> None:
    log_dir = "././01_data/logs"
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
    logging.info(f"[START] 크롤링 시작: {week_str}")


# ==========================================
# 2) HTTP 재시도
# ==========================================
def fetch_with_retry(url: str, max_retries: int = 3, backoff: int = 2) -> requests.Response:
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait = backoff ** attempt
                logging.warning(f"[WARNING] 요청 실패, {wait}초 후 재시도 → {url}")
                time.sleep(wait)
            else:
                logging.error(f"[FAILED] 최대 재시도 초과: {url}")
                raise


# ==========================================
# 3) 주차 URL 생성
# ==========================================
def get_week_url(year: int = None, week: int = None):
    now = datetime.now()
    year = year or now.year
    week = week or now.isocalendar()[1]

    week_str = f"{year}-W{week:02d}"
    url = f"https://huggingface.co/papers/week/{week_str}"
    return week_str, url


# ==========================================
# 4) 논문 목록 크롤링
# ==========================================
def fetch_paper_urls(weekly_url: str) -> List[Dict[str, str]]:
    response = fetch_with_retry(weekly_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    papers = []
    for link in soup.select('a.line-clamp-3'):
        href = link.get("href")
        title = link.text.strip()
        if href:
            papers.append({
                "title": title,
                "url": f"https://huggingface.co{href}"
            })

    logging.info(f"[COUNT] 총 {len(papers)}개 논문 발견")
    return papers


# ==========================================
# 5) 논문 상세 정보
# ==========================================
def fetch_paper_details(paper_url: str, paper_title: str):
    try:
        response = fetch_with_retry(paper_url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Abstract
        abstract = ""
        section = soup.select_one('section div')
        if section:
            paragraphs = section.find_all('p')
            abstract = " ".join([p.get_text(strip=True) for p in paragraphs])

        if not abstract or len(abstract) < 50:
            logging.warning(f"[WARNING] Abstract 너무 짧음: {paper_title}")
            return None

        # GitHub URL
        gh = soup.select_one('a[href*="github.com"]')
        github_url = gh['href'] if gh else ""

        # Upvote
        upvote_elem = soup.select_one('div.font-semibold.text-orange-500')
        text = upvote_elem.text.strip() if upvote_elem else "-"
        upvote = 0 if text == "-" else int("".join(filter(str.isdigit, text)))

        return {
            "title": paper_title,
            "abstract": abstract,
            "github_url": github_url,
            "huggingface_url": paper_url,
            "upvote": upvote
        }

    except Exception as e:
        logging.error(f"[FAILED] 상세 정보 실패: {e}")
        return None


# ==========================================
# 6) 🔥 TF-IDF + NLTK 키워드 추출 (새 버전)
# ==========================================
def extract_keywords_tfidf_nltk(text: str, top_n: int = 3) -> List[str]:

    if not text or len(text.split()) < 10:
        return [f"keyword{i+1}" for i in range(top_n)]

    try:
        lemmatizer = WordNetLemmatizer()
        nltk_stop = set(stopwords.words("english"))

        custom_stop = {
            "paper", "propose", "proposed", "proposes",
            "show", "shows", "using", "used", "approach",
            "method", "model", "models", "based", "work",
            "task", "tasks", "result", "results", "data",
            "we", "our", "they", "their", "its", "this"
        }

        stop_words = nltk_stop.union(custom_stop)

        # Tokenize
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
        tokens = [tok for tok in tokens if tok not in stop_words]

        processed = " ".join(tokens)
        if not processed.strip():
            return [f"keyword{i+1}" for i in range(top_n)]

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            lowercase=False,
            token_pattern=r"\b[a-z]{3,}\b"
        )

        tfidf_matrix = vectorizer.fit_transform([processed])
        features = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        idx = scores.argsort()[::-1][:top_n]
        raw_keywords = [features[i] for i in idx]

        # 긴 키워드를 우선시키는 중복 제거
        sorted_kw = sorted(raw_keywords, key=len, reverse=True)
        filtered = []

        for kw in sorted_kw:
            if not any(kw in longer for longer in filtered):
                filtered.append(kw)

        keywords = [kw for kw in raw_keywords if kw in filtered]

        while len(keywords) < top_n:
            keywords.append(f"keyword{len(keywords)+1}")

        return keywords[:top_n]

    except Exception as e:
        logging.error(f"[ERROR] TF-IDF 키워드 실패: {e}")
        return [f"keyword{i+1}" for i in range(top_n)]


# ==========================================
# 7) JSON 저장
# ==========================================
def save_document_json(data, week_str, index):
    year = week_str[:4]
    week = week_str.split("-W")[1]

    doc_id = f"doc{year[2:]}{week}{index+1:03d}"
    save_dir = f"././01_data/documents/{year}/{week_str}"
    os.makedirs(save_dir, exist_ok=True)

    filepath = f"{save_dir}/{doc_id}.json"

    document = {
        "context": data['abstract'],
        "metadata": {
            "paper_name": data['title'],
            "github_url": data['github_url'],
            "huggingface_url": data['huggingface_url'],
            "upvote": data['upvote'],
            "tags": data['tags'],
        }
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    return doc_id, filepath


# ==========================================
# 8) CSV 저장
# ==========================================
def save_metadata_csv(saved, week_str):
    year = week_str[:4]
    csv_path = f"././01_data/documents/{year}/{week_str}/docs_info.csv"

    rows = []
    for p in saved:
        rows.append({
            "doc_id": p["doc_id"],
            "paper_name": p["metadata"]["paper_name"],
            "doc_file": f"{p['doc_id']}.json",
            "github_url": p["metadata"]["github_url"],
            "huggingface_url": p["metadata"]["huggingface_url"],
            "upvote": p["metadata"]["upvote"],
            "tags": json.dumps(p["metadata"]["tags"], ensure_ascii=False)
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


# ==========================================
# 9) 메인 크롤러
# ==========================================
def crawl_weekly_papers(year=None, week=None):

    week_str, url = get_week_url(year, week)
    setup_logging(week_str)

    links = fetch_paper_urls(url)
    saved_papers = []

    for idx, paper in enumerate(links):
        logging.info(f"처리 중 {idx+1}/{len(links)} → {paper['title'][:50]}")

        details = fetch_paper_details(paper["url"], paper["title"])
        if not details:
            continue

        # 🔥 여기서 새로운 키워드 추출 함수 사용
        keywords = extract_keywords_tfidf_nltk(details["abstract"], top_n=3)
        details["tags"] = keywords

        doc_id, _ = save_document_json(details, week_str, idx)

        saved_papers.append({
            "doc_id": doc_id,
            "context": details["abstract"],
            "metadata": {
                "paper_name": details["title"],
                "github_url": details["github_url"],
                "huggingface_url": details["huggingface_url"],
                "upvote": details["upvote"],
                "tags": keywords
            }
        })

        time.sleep(1)

    if saved_papers:
        save_metadata_csv(saved_papers, week_str)

    logging.info("[SUCCESS] 주간 크롤링 완료")


# ==========================================
# 실행
# ==========================================
if __name__ == "__main__":
    crawl_weekly_papers(year=2025, week=49)
