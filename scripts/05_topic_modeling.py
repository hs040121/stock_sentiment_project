import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# =======================================
# 경로 설정 (절대 경로 기반)
# =======================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "results", "topic_modeling")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📁 토픽 저장 폴더:", OUTPUT_DIR)

INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "naver_board_kospi100_with_sentiment.csv")

# =======================================
# 토픽 모델링 시작
# =======================================
def main():
    print("데이터 로드 중...")
    df = pd.read_csv(INPUT_PATH, encoding="utf-8")
    print("총 종목 수:", df["종목명"].nunique())

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    for ticker in df["종목명"].unique():

        print(f"\n=== {ticker} 토픽 모델링 중 ===")
        sub = df[df["종목명"] == ticker]

        if len(sub) < 20:
            print(f" ⛔ {ticker} 데이터 부족 → 스킵")
            continue

        docs = sub["제목_전처리"].tolist()
        embeddings = model.encode(docs, show_progress_bar=False)

        topic_model = BERTopic(language="multilingual")
        topics, probs = topic_model.fit_transform(docs, embeddings)

        topic_info = topic_model.get_topic_info()
        documents = topic_model.get_document_info(docs)

        # 저장 경로
        save_path_topics = os.path.join(OUTPUT_DIR, f"{ticker}_topics.csv")
        save_path_docs = os.path.join(OUTPUT_DIR, f"{ticker}_docs.csv")

        topic_info.to_csv(save_path_topics, index=False, encoding="utf-8-sig")
        documents.to_csv(save_path_docs, index=False, encoding="utf-8-sig")

        print(f" ✔ 저장 완료 → {save_path_topics}")

    print("\n🎉 모든 종목 토픽 모델링 완료!")

if __name__ == "__main__":
    main()
