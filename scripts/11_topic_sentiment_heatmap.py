import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# ==========================
# 설정
# ==========================
INPUT_PATH = "../data/naver_board_kospi100_with_sentiment.csv"
OUT_DIR = "../results/topic_sentiment_heatmap"
os.makedirs(OUT_DIR, exist_ok=True)

# 히트맵에 넣을 종목 수/토픽 수(너무 크면 보기 지저분)
TOP_TICKERS = 10          # 댓글 많은 종목 기준 상위 N개
TOPICS_PER_TICKER = 8     # 각 종목에서 표시할 토픽 수(빈도 상위)
MIN_DOCS_TICKER = 80      # 종목별 최소 문서 수(적으면 스킵)

plt.rcParams["axes.unicode_minus"] = False
try:
    plt.rcParams["font.family"] = "Malgun Gothic"
except:
    pass

def safe_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", str(name))
    return name[:80]

def find_sentiment_col(df):
    cand = [c for c in df.columns if "sentiment" in c.lower()]
    if cand:
        return cand[-1]
    if "label" in df.columns:
        return "label"
    raise KeyError("❌ sentiment 컬럼을 찾을 수 없음")

def normalize_sentiment(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    uniq = set(pd.unique(s.dropna()))
    if uniq.issubset({0, 1}):
        s = s.map({0: -1, 1: 1})
    return s

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def plot_heatmap(mat: pd.DataFrame, title: str, filename: str):
    # mat: index=topic_label, columns=ticker
    plt.figure(figsize=(max(10, 1.2*len(mat.columns)), max(6, 0.5*len(mat.index))))
    plt.imshow(mat.values, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.colorbar()
    save_fig(os.path.join(OUT_DIR, filename))

def main():
    df = pd.read_csv(INPUT_PATH, encoding="utf-8")
    if "종목명" not in df.columns:
        raise KeyError("❌ '종목명' 컬럼이 없습니다.")
    if "제목_전처리" not in df.columns:
        raise KeyError("❌ '제목_전처리' 컬럼이 없습니다.")

    sent_col = find_sentiment_col(df)
    df["_sent"] = normalize_sentiment(df[sent_col])
    df = df.dropna(subset=["_sent"]).copy()

    # 분석할 종목 선택: 댓글 수 많은 TOP N
    ticker_counts = df["종목명"].value_counts()
    tickers = [t for t in ticker_counts.head(TOP_TICKERS).index if ticker_counts[t] >= MIN_DOCS_TICKER]

    print("분석 종목:", tickers)
    if len(tickers) == 0:
        raise ValueError("❌ 조건(MIN_DOCS_TICKER 등) 때문에 분석할 종목이 없습니다.")

    embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 히트맵용 행을 만들기 위해 “TopicLabel”을 통일된 형태로 만들자:
    # 예) "T0(실적/호재)" 같은 문자열
    all_rows_score = []   # 평균 감성(-1~1)
    all_rows_pos = []     # 긍정비율(0~1)
    all_topic_tables = [] # 토픽 요약 테이블(보고서/부록용)

    for ticker in tickers:
        sub = df[df["종목명"] == ticker].copy()
        if len(sub) < MIN_DOCS_TICKER:
            continue

        docs = sub["제목_전처리"].astype(str).tolist()
        sent = sub["_sent"].astype(int).tolist()

        print(f"\n=== {ticker} BERTopic 학습 중 (n={len(docs)}) ===")
        embeddings = embed_model.encode(docs, show_progress_bar=False)

        topic_model = BERTopic(language="multilingual")
        topics, probs = topic_model.fit_transform(docs, embeddings)

        tmp = pd.DataFrame({
            "doc": docs,
            "sent": sent,
            "topic": topics
        })

        # -1 토픽(아웃라이어)은 제외하면 보기 좋아짐
        tmp = tmp[tmp["topic"] != -1].copy()
        if len(tmp) == 0:
            print("  ⛔ 유효 토픽이 거의 없어 스킵")
            continue

        # 토픽별 통계
        agg = tmp.groupby("topic").agg(
            n=("sent", "size"),
            mean_sent=("sent", "mean"),
            pos_ratio=("sent", lambda x: (x == 1).mean())
        ).reset_index()

        # 빈도 상위 TOPICS_PER_TICKER만 사용
        agg = agg.sort_values("n", ascending=False).head(TOPICS_PER_TICKER).copy()

        # 토픽 키워드 추출해서 라벨 생성
        topic_labels = []
        top_words_list = []
        for tid in agg["topic"].tolist():
            words = [w for (w, _) in topic_model.get_topic(tid)][:5]
            top_words_list.append(", ".join(words))
            # 짧게 2개 단어만 라벨에
            short = "/".join(words[:2]) if len(words) >= 2 else (words[0] if words else "topic")
            topic_labels.append(f"{ticker} | T{tid}({short})")

        agg["topic_label"] = topic_labels
        agg["top_words"] = top_words_list
        agg["ticker"] = ticker

        all_topic_tables.append(agg[["ticker", "topic", "topic_label", "n", "mean_sent", "pos_ratio", "top_words"]])

        # 히트맵용 row 구성
        for _, r in agg.iterrows():
            all_rows_score.append({
                "topic_label": r["topic_label"],
                "ticker": ticker,
                "value": float(r["mean_sent"])
            })
            all_rows_pos.append({
                "topic_label": r["topic_label"],
                "ticker": ticker,
                "value": float(r["pos_ratio"])
            })

        # 종목별 토픽 요약 CSV 저장
        out_csv = os.path.join(OUT_DIR, f"{safe_filename(ticker)}_topic_sentiment_table.csv")
        agg.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("  ✅ 저장:", out_csv)

    # ==========================
    # 전체 히트맵 만들기
    # ==========================
    score_df = pd.DataFrame(all_rows_score)
    pos_df = pd.DataFrame(all_rows_pos)

    if len(score_df) == 0:
        raise ValueError("❌ 히트맵을 만들 데이터가 없습니다. (토픽 생성 실패/스킵)")

    score_mat = score_df.pivot_table(index="topic_label", columns="ticker", values="value", aggfunc="mean")
    pos_mat = pos_df.pivot_table(index="topic_label", columns="ticker", values="value", aggfunc="mean")

    # NaN은 0으로 채워서 표시(해당 종목에 없는 토픽)
    score_mat = score_mat.fillna(0)
    pos_mat = pos_mat.fillna(0)

    plot_heatmap(score_mat, "Topic × 평균 감성(Mean Sentiment)", "01_heatmap_topic_mean_sent.png")
    plot_heatmap(pos_mat, "Topic × 긍정비율(Pos Ratio)", "02_heatmap_topic_pos_ratio.png")

    # ==========================
    # 전체 토픽 테이블 합치기(부록용)
    # ==========================
    full_table = pd.concat(all_topic_tables, axis=0, ignore_index=True)
    full_out = os.path.join(OUT_DIR, "topic_sentiment_full_table.csv")
    full_table.to_csv(full_out, index=False, encoding="utf-8-sig")
    print("\n✅ 전체 토픽-감성 테이블 저장:", full_out)

    # 추가: 평균 감성 TOP/BOTTOM 15 저장
    top15 = full_table.sort_values("mean_sent", ascending=False).head(15)
    bot15 = full_table.sort_values("mean_sent", ascending=True).head(15)
    top15.to_csv(os.path.join(OUT_DIR, "top15_topics_by_mean_sent.csv"), index=False, encoding="utf-8-sig")
    bot15.to_csv(os.path.join(OUT_DIR, "bottom15_topics_by_mean_sent.csv"), index=False, encoding="utf-8-sig")
    print("✅ TOP/BOTTOM 토픽 CSV 저장 완료")

    print("\n🎉 완료! 결과 폴더:", OUT_DIR)

if __name__ == "__main__":
    main()
