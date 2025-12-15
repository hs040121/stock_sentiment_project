# 📊 KoELECTRA 기반 주식 종목별 투자자 감성 및 토픽–감성 결합 분석

본 프로젝트는 네이버 증권 종목 토론 게시판 데이터를 활용하여  
**KoELECTRA 기반 감성 분석**과 **BERTopic 기반 토픽 모델링**을 결합함으로써,  
주식 종목별 투자자 담론의 구조적 특성을 분석하는 것을 목표로 한다.

---

## 📌 프로젝트 개요

- **주제**: 주식 종목별 투자자 감성 분석 및 토픽–감성 결합 분석  
- **데이터 출처**: 네이버 증권 종목 토론 게시판  
- **대상 종목**: KOSPI100 주요 종목  
- **분석 단위**: 게시글 제목  
- **핵심 기법**:
  - KoELECTRA 기반 이진 감성 분류
  - BERTopic 기반 토픽 모델링
  - 토픽–감성 결합 히트맵 분석

---

## 🎯 연구 목적

기존의 감성 분석은 종목 단위 평균 감성 지표에 의존하는 경우가 많아,  
**투자자 감성이 어떤 이슈(토픽)에 의해 형성되는지**를 설명하는 데 한계가 있다.

본 프로젝트는 다음 질문에 답하고자 한다.

- 종목별 투자자 감성은 어떤 토픽에 의해 형성되는가?
- 동일 종목 내에서도 토픽에 따라 감성 분포는 어떻게 달라지는가?

---

## 🗂 데이터 구성 및 전처리

- 수집 항목: `종목명`, `게시글 제목`
- 전처리 과정:
  - 특수문자 및 불필요한 기호 제거
  - 한글 외 문자 정리
  - 중복 및 의미 없는 문장 제거
- 최종 데이터 수: **17,477건**

> ⚠️ 데이터 저작권 문제로 원본 데이터는 GitHub에 직접 업로드하지 않고,  
> 수집 및 전처리 코드를 공개하였다.

---

## 🤖 감성 분석 모델

### 라벨링 전략
- 긍정(1): 상승 기대, 호재, 긍정적 전망
- 부정(-1): 하락 우려, 악재, 비관적 전망
- 키워드 기반 약지도 방식으로 초기 라벨링
- 클래스 불균형 완화를 위해 **균형 데이터셋 구성**

### 모델 정보
- **모델**: KoELECTRA-base
- **프레임워크**: PyTorch, HuggingFace Transformers
- **분류 방식**: 이진 분류 (Positive / Negative)

---

## 📈 기본 분석 결과

### 전체 감성 분포
> 전체 데이터에서 긍정 및 부정 감성이 비교적 균형 잡힌 분포를 보임

📌 **[그림 위치]**  
- `results/figures_clean/01_overall_sentiment.png`

<img width="1540" height="1100" alt="01_overall_sentiment" src="https://github.com/user-attachments/assets/225e22eb-523f-4fc8-bd81-e2cf9d48bfd9" />

---

### 종목별 감성 비교
- 종목별 평균 감성 점수 및 긍정 비율 비교
- 종목 간 투자자 감성 차이가 뚜렷하게 나타남

📌 **[그림 위치]**  
- `results/figures_clean/03_score_top10.png`  

<img width="2200" height="1320" alt="03_score_top10" src="https://github.com/user-attachments/assets/a5959b59-fe04-47b4-a371-b6cdc6c4d5ff" />

- `results/figures_clean/04_score_bottom10.png`  

<img width="2200" height="1320" alt="04_score_bottom10" src="https://github.com/user-attachments/assets/84997a4a-a309-45bc-ae90-d8ce3f59f65d" />

---

### TOP10 종목 긍·부정 비율 비교
종목별 평균 감성 점수 비교를 보완하기 위해,  
긍정 비율 기준 상위 10개 종목을 대상으로 긍·부정 비율을 비교하였다.

📌 **[그림 위치]**  
- `results/figures_clean/05_top10_pos_neg_ratio.png`

<!-- 아래 src에 GitHub 이미지 URL만 넣으면 됨 -->
<img width="2200" height="1320" alt="05_top10_pos_neg_ratio" src="여기에_05번_이미지_URL_붙여넣기" />
<img width="2200" height="1320" alt="05_top10_pos_neg_ratio" src="https://github.com/user-attachments/assets/a62b09cb-a019-4184-8011-7a1f3629230f" />

#### 해석
대부분의 상위 종목에서 **긍정 비율이 부정보다 높게 나타났으며**,  
이는 해당 종목들에 대해 전반적으로 우호적인 투자자 심리가 형성되어 있음을 시사한다.  
다만, 본 분석은 종목 단위 집계 결과이므로 개별 이슈나 담론의 성격을 설명하는 데에는 한계가 있다.  
이에 따라 다음 절에서는 **토픽 모델링과 감성 분석을 결합한 분석**을 수행하였다.

---

## 🧠 토픽 모델링

- **토픽 모델**: BERTopic
- **임베딩 모델**: Sentence-BERT (multilingual)
- 아웃라이어 토픽(-1)은 분석에서 제외

### 분석 대상 종목 선정 기준
1. 댓글 수 상위 종목
2. 종목별 최소 문서 수 80개 이상
3. 유효 토픽이 충분히 생성된 종목만 분석

---

## 🔍 토픽–감성 결합 분석 (확장 분석)

### 분석 방법
- 종목별 주요 토픽에 대해 다음 지표 계산:
  - 평균 감성 점수 (Mean Sentiment)
  - 긍정 비율 (Positive Ratio)
- 결과를 히트맵(Heatmap) 형태로 시각화

📌 **[그림 위치]**
- `results/topic_sentiment_heatmap/01_heatmap_topic_mean_sent.png`
- `results/topic_sentiment_heatmap/02_heatmap_topic_pos_ratio.png`

<!-- 원하면 아래처럼 이미지도 붙이면 됨 (URL 채우기)
<img width="2200" height="1320" alt="01_heatmap_topic_mean_sent" src="여기에_히트맵1_URL" />
<img width="2200" height="1320" alt="02_heatmap_topic_pos_ratio" src="여기에_히트맵2_URL" />
-->

---

### 주요 분석 결과
- 동일 종목 내에서도 **토픽별 감성 편차가 큼**
- 부정적인 종목은 **특정 부정 토픽에 감성이 집중**
- 긍정적인 종목은 **여러 토픽에서 고른 긍정 분포**

> 이를 통해 단순 감성 평균을 넘어,  
> **투자자 감성이 형성되는 구조적 요인**을 확인할 수 있었다.

---

## 🧾 프로젝트 구조

