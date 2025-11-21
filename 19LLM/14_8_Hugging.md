1️⃣ 모델 클래스 종류와 특징
- Hugging Face Transformers에서는 모델 구조와 목적에 따라 모델 클래스가 다르게 나뉩니다.

| 용도                | 대표 모델           | 모델 클래스                          | 추천 토크나이저                                        | 비고                       |
| ----------------- | --------------- | ------------------------------- | ----------------------------------------------- | ------------------------ |
| 자유로운 텍스트 생성       | GPT2            | `GPT2LMHeadModel`               | `GPT2TokenizerFast` / `PreTrainedTokenizerFast` | Autoregressive LM        |
| 자유로운 텍스트 생성       | GPT-Neo / GPT-J | `AutoModelForCausalLM`          | `AutoTokenizer`                                 | 큰 모델일수록 AutoTokenizer 편리 |
| 요약 / 번역 / 조건부 생성  | BART (KoBART)   | `BartForConditionalGeneration`  | `PreTrainedTokenizerFast` / `AutoTokenizer`     | Seq2Seq LM               |
| 요약 / 번역 / 조건부 생성  | T5              | `T5ForConditionalGeneration`    | `T5Tokenizer` / `AutoTokenizer`                 | 입력 프롬프트 설계 중요            |
| 문장 분류 / 감정 분석     | BERT            | `BertForSequenceClassification` | `BertTokenizerFast` / `AutoTokenizer`           | 분류용 특화                   |
| 토큰 분류 / NER       | RoBERTa         | `RobertaForTokenClassification` | `RobertaTokenizerFast` / `AutoTokenizer`        | 토큰 단위 라벨링용               |
| 마스크된 단어 예측 / 사전학습 | ELECTRA         | `ElectraForMaskedLM`            | `ElectraTokenizerFast` / `AutoTokenizer`        | MLM 사전학습 모델              |



# Hugging Face 모델 & 토크나이저 Cheat Sheet

---

## 1️⃣ 모델 구조별 요약

### Autoregressive 모델 (GPT2, GPT-Neo 등)
- **설명:** 앞에서부터 다음 단어 예측  
- **용도:** 자유로운 텍스트 생성, 채팅, 글쓰기  
- **모델 클래스:** `GPT2LMHeadModel`, `AutoModelForCausalLM`  
- **토크나이저:** `GPT2TokenizerFast`, `PreTrainedTokenizerFast`  

**예제 코드:**
```python
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import torch

tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2')
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

text = "안녕하세요, 오늘 날씨는"
input_ids = tokenizer.encode(text, return_tensors='pt')

output_ids = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```



### Seq2Seq 모델 (BART, T5)

- **설명:** 입력 문장 → 출력 문장 구조  
- **용도:** 요약, 번역, 질문-응답 등 조건부 생성  
- **모델 클래스:** `BartForConditionalGeneration`, `T5ForConditionalGeneration`, `AutoModelForSeq2SeqLM`  
- **토크나이저:** `PreTrainedTokenizerFast`, `AutoTokenizer`  

**예제 코드 (KoBART 요약):**
```python
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import torch

tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')

text = "과거를 떠올려보자. 방송을 보던 우리의 모습을..."
input_ids = tokenizer.encode(text, return_tensors='pt')

summary_ids = model.generate(input_ids, max_length=50)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```


## 분류 / 토큰화 특화 모델

- **설명:** 입력 문장 분석용  
- **모델 클래스:** `BertForSequenceClassification`, `RobertaForTokenClassification` 등  
- **토크나이저:** 모델 구조에 맞는 Fast 토크나이저 또는 `AutoTokenizer`  

**예제 코드 (BERT 문장 분류):**
```python
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

text = "I love this movie!"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
print(outputs.logits)
```

## 2️⃣ 실무 팁

- **모델 이름만 아는 경우:**  
  `AutoModelForSeq2SeqLM` 또는 `AutoModelForCausalLM` 사용
- **세부 조정/학습이 필요한 경우:**  
  해당 구조 전용 모델 (`BartForConditionalGeneration`, `GPT2LMHeadModel`) 사용
- **용도에 맞춰 선택:**  
  - 글 생성 → GPT2, GPT-Neo  
  - 요약/번역 → BART, T5  
  - 분류 → BERT, RoBERTa

---

## 3️⃣ 추가 실무 팁

- **모델 용도 → 구조 → 토크나이저 순서로 선택**  
  예: 요약 → Seq2Seq → BART → `PreTrainedTokenizerFast`

- **Auto~ 시리즈:**  
  - 모델 이름만 있으면 자동으로 맞는 토크나이저/모델을 불러줌  
  - 초보자/빠른 실험용 추천

- **PreTrainedTokenizerFast:**  
  - 속도 빠르고 세부 옵션 조정 가능  
  - 직접 학습/미세조정할 때 유리

- **GPT 계열:**  
  - 프롬프트 설계가 성능에 큰 영향  
  - 자유 생성, 채팅, 글쓰기 용도

- **Seq2Seq 계열:**  
  - 입력 → 출력 구조로 조건부 생성  
  - 요약, 번역, 질의응답 등에 최적화


<br>

| 모델 유형                       | 구조 이름(Architecture) | 대표 모델(Model Examples)                        | 용도             | 흐름 요약                                                                               |
| --------------------------- | ------------------- | -------------------------------------------- | -------------- | ----------------------------------------------------------------------------------- |
| **Seq2Seq**                 | Encoder–Decoder     | BART, T5, Pegasus, mBART, FLAN-T5            | 요약, 번역, 조건부 생성 | [입력 문장] → 토크나이즈 → [인코더] → 숨은 상태(hidden context) → [디코더] → 다음 토큰 예측 반복 → 토큰 ID → 디코딩 |
| **Autoregressive (GPT 계열)** | Decoder-only        | GPT-2, GPT-Neo, GPT-J, LLaMA, Mixtral, Gemma | 자유 생성, 채팅, 글쓰기 | [프롬프트] → 토크나이즈 → [디코더] → 다음 토큰 예측 반복 → 토큰 ID → 디코딩                                  |
| **Encoder-only (BERT 계열)**  | Encoder-only        | BERT, RoBERTa, ELECTRA, DistilBERT           | 분류, QA, 태깅     | [입력 문장] → 토크나이즈 → [인코더] → hidden states → [CLS] → 선형 레이어 → 로짓 → softmax → 클래스 예측    |



<br>

- 토크나이즈가 항상 먼저
- 인코더와 디코더는 모델 내부에서 연결
- 대부분 model.generate()를 호출하면 모델 내부에서 인코더와 디코더를 자동으로 연결되어 처리
- [입력 문장] --> [인코더] --> [컨텍스트 벡터 / 숨은 상태] --> [디코더] --> [출력 문장]

<br>
<br>

| 디코딩 방식                       | 종류                 | 어떻게 작동하나(설명)                                          | 장점                                | 단점                                  | 언제 사용하면 좋은가                                      |
| ---------------------------- | ------------------ | ----------------------------------------------------- | --------------------------------- | ----------------------------------- | ------------------------------------------------ |
| **Greedy Search**            | 확정적(deterministic) | 매 단계에서 **가장 확률 높은 토큰을 1개 선택**                         | 빠름, 간단함                           | 반복적/단조롭고 품질 낮을 수 있음, 전역 최적 해 찾기 어려움 | 규칙적/예측 가능한 답이 필요할 때, 짧은 응답                       |
| **Beam Search**              | 확정적(deterministic) | 매 단계에서 **상위 B개의 후보 문장(beam)** 유지하며 확률 조합이 가장 높은 문장 선택 | Greedy보다 훨씬 고품질, 특히 번역·요약 성능 좋음   | 느림, beam이 너무 크면 부자연스러운 결과도          | 번역, 요약, 정형화된 생성 작업                               |
| **Top-k Sampling**           | 확률적(stochastic)    | 확률 상위 **k개 후보 중 랜덤 샘플링**                              | 다양성 증가, 창의적                       | k 선택이 까다롭고, 비현실적 단어 나올 위험 있음        | 창의적 글쓰기, 소설/스토리 생성                               |
| **Top-p (Nucleus) Sampling** | 확률적(stochastic)    | 누적 확률이 **p(예: 0.9)**가 될 때까지 후보를 모아 그 안에서 랜덤 선택        | Top-k보다 더 자연스럽고 안정적, 다양성+품질 균형 좋음 | 여전히 랜덤성 존재 → 일관성 약할 수 있음            | 대부분의 생성 작업, 창의적이면서도 품질 유지 필요할 때 (GPT 기본 설정에 가까움) |


<br>
- 모델 구조를 알려주는 코드
