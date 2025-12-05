import os
import time
import random
import re
import json
from collections import Counter
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import logging
from datetime import datetime

# =========================
# TF-IDF + Lemmatizer 기반 태그 추출
# =========================
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List

# NLTK 다운로드
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def extract_keywords_tfidf_nltk(text: str, top_n: int = 3) -> List[str]:
    if not text or len(text.split()) < 10:
        logging.warning("[WARNING] Abstract가 너무 짧음 (<10 words), 기본 키워드 반환")
        return [f"keyword{i+1}" for i in range(top_n)]

    try:
        lemmatizer = WordNetLemmatizer()

        nltk_stopwords = set(stopwords.words('english'))
        custom_stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "been", "be",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "this", "that", "these", "those",
            "we", "our", "their", "they", "it", "its", "which", "who", "when",
            "where", "why", "how", "what", "if", "than", "such", "into", "through",

            # 논문에서 너무 많이 나오는 단어들
            "paper", "propose", "present", "show", "demonstrate", "using", "used",
            "approach", "method", "model", "based", "results", "work",
            "task", "tasks", "result", "results", "data"
        }
        all_stopwords = nltk_stopwords.union(custom_stopwords)

        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        lemmatized_tokens = [lemmatizer.lemmatize(t) for t in tokens]
        filtered_tokens = [t for t in lemmatized_tokens if t not in all_stopwords]

        preprocessed_text = ' '.join(filtered_tokens)
        if not preprocessed_text.strip():
            logging.warning("[WARNING] 전처리 후 텍스트 비어있음 → 기본 키워드 반환")
            return [f"keyword{i+1}" for i in range(top_n)]

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            lowercase=False,
            token_pattern=r'\b[a-z]{3,}\b'
        )
        tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        idx_score_pairs = [(i, s) for i, s in enumerate(scores) if s > 0]
        if not idx_score_pairs:
            return [f"keyword{i+1}" for i in range(top_n)]

        idx_score_pairs.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in idx_score_pairs[:top_n]]
        keywords = [feature_names[i] for i in top_indices]

        # 중복 제거
        filtered = []
        for kw in sorted(keywords, key=len, reverse=True):
            if not any(kw in other for other in filtered):
                filtered.append(kw)

        while len(filtered) < top_n:
            filtered.append(f"keyword{len(filtered)+1}")

        return filtered[:top_n]

    except Exception as e:
        logging.error(f"[ERROR] TF-IDF 처리 실패: {e}")
        return [f"keyword{i+1}" for i in range(top_n)]


# ====== 설정 ======
base_year = 2025
start_week = 45
wait_time = 7
max_retry_per_article = 4
retry_click = 6

# ====== 로깅 설정 ======
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_week_str = f"{base_year}-W{start_week:02d}"
log_file = f"crawling_{log_week_str}_{current_time}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info(f"🚀 크롤링 시작 — 로그파일: {log_file}")

# ====== 웹 드라이버 실행 ======
options = webdriver.ChromeOptions()
# options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("user-agent=Mozilla/5.0")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

week = start_week
week_url = f"https://huggingface.co/papers/week/{base_year}-W{week:02d}"
file_index = int(str(base_year)[-2:] + f"{week:02d}" + "001")

total_articles = 0
success_count = 0
fail_count = 0

# ====== 크롤링 루프 ======
while True:
    logging.info(f"🔹 Crawling week URL: {week_url}")
    folder = f"{base_year}-W{week:02d}"
    os.makedirs(folder, exist_ok=True)
    time.sleep(random.uniform(2, 4))

    try:
        driver.get(week_url)
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article h3 a"))
        )
        articles = driver.find_elements(By.CSS_SELECTOR, "article h3 a")
        article_urls = [a.get_attribute("href") for a in articles]
        logging.info(f"📝 {len(article_urls)} articles found")
    except Exception as e:
        logging.error(f"❌ No articles found or page error: {e}")
        break

    total_articles += len(article_urls)

    for link in article_urls:
        article_success = False

        for attempt in range(1, max_retry_per_article + 1):
            try:
                driver.get(link)
                time.sleep(random.uniform(3, 6))

                # 제목
                try:
                    paper_name = WebDriverWait(driver, wait_time).until(
                        EC.presence_of_element_located((By.TAG_NAME, "h1"))
                    ).text.strip()
                except:
                    paper_name = "Unknown_Title"

                # Abstract
                try:
                    abstract_div = WebDriverWait(driver, wait_time).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "div.pb-8.pr-4.md\\:pr-16 > div")
                        )
                    )
                    ps = abstract_div.find_elements(By.TAG_NAME, "p")
                    page_content = "\n".join([p.text.strip() for p in ps]) if ps else abstract_div.text.strip()
                except:
                    page_content = ""

                # Upvote
                try:
                    upvote_elem = WebDriverWait(driver, wait_time).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR,
                            "section.pt-8 div.hidden.flex-wrap.items-start.gap-2.md\\:flex a div div"
                        ))
                    )
                    upvote_match = re.search(r"\d+", upvote_elem.text.strip())
                    upvote = int(upvote_match.group()) if upvote_match else 0
                except:
                    upvote = 0

                # Github 링크
                try:
                    github_url = driver.find_element(By.XPATH, "//a[contains(@href,'github.com')]").get_attribute("href")
                except:
                    github_url = ""

                huggingface_url = link

                # ** TF-IDF 기반 태그 생성 **
                tags = extract_keywords_tfidf_nltk(page_content, top_n=3)

                json_data = {
                    "content": page_content,
                    "metadata": {
                        "paper_name": paper_name,
                        "github_url": github_url,
                        "huggingface_url": huggingface_url,
                        "upvote": upvote,
                        "tags": tags
                    }
                }

                doc_name = f"doc{file_index}.json"
                file_path = os.path.join(folder, doc_name)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4)

                logging.info(f"✅ Saved {file_path}")
                file_index += 1
                success_count += 1
                article_success = True
                break

            except Exception as e:
                logging.warning(f"⚠️ Retry {attempt}/{max_retry_per_article} failed for {link}, error: {e}")
                time.sleep(3)

        if not article_success:
            logging.error(f"❌ Failed to crawl article: {link}")
            fail_count += 1

    clicked = False
    for attempt in range(retry_click):
        try:
            driver.get(week_url)
            next_btn = WebDriverWait(driver, wait_time).until(
                EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/main/div[2]/section/div[1]/div[4]/div/div[2]/a[2]"))
            )
            next_btn.click()
            time.sleep(random.uniform(3, 6))
            week += 1
            week_url = driver.current_url
            file_index = int(str(base_year)[-2:] + f"{week:02d}" + "001")
            clicked = True
            logging.info(f"➡ Moving to next week: {week_url}")
            break
        except:
            logging.warning(f"⚠️ Next button click attempt {attempt+1}/{retry_click} failed")
            time.sleep(2)

    if not clicked:
        logging.info("➡ No more weeks. Crawling finished.")
        break

driver.quit()

logging.info("🎉 크롤링 완료!")
logging.info(f"총 아티클 수: {total_articles}")
logging.info(f"성공: {success_count}")
logging.info(f"실패: {fail_count}")
