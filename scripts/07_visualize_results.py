# scripts/07_visualize_results.py
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# 경로 설정
# ==========================
INPUT_PATH = "../data/naver_board_kospi100_with_sentiment.csv"
OUT_DIR = "../results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["axes.unicode_minus"] = False  # 음수 깨짐 방지
# Windows 한글 폰트(있는 경우)
try:
    plt.rcParams["font.family"] = "Malgun Gothic"
except:
    pass

# ==========================
# 유틸
# ==========================
def safe_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", str(name))
    return name[:80]

def find_sentiment_col(df: pd.DataFrame) -> str:
    # sentiment 들어간 컬럼 우선 탐지
    cand = [c for c in df.columns if "sentiment" in c.lower()]
    if cand:
        return cand[-1]  # 가장 마지막 컬럼을 최신으로 가정
    # 혹시 label 컬럼만 있는 경우
    if "label" in df.columns:
        return "label"
    raise KeyError("❌ sentiment 관련 컬럼을 찾을 수 없음 (예: sentiment_binary, label)")

def save_fig(filename: str):
    path = os.path.join(OUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    print("✅ 저장:", path)
    plt.close()

def value_counts_plot(series: pd.Series, title: str, filename: str, xlabel: str = "", ylabel: str = "count"):
    vc = series.value_counts(dropna=False)
    plt.figure(figsize=(8,5))
    plt.bar(vc.index.astype(str), vc.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i, v in enumerate(vc.values):
        plt.text(i, v + (max(vc.values)*0.02 if max(vc.values) > 0 else 1), str(v), ha="center")
    save_fig(filename)

# ==========================
# 메인
# ==========================
def main():
    print("데이터 로드:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH, encoding="utf-8")
    print("행:", len(df), "컬럼:", len(df.columns))

    # 필수 컬럼 확인
    if "종목명" not in df.columns:
        raise KeyError("❌ '종목명' 컬럼이 없습니다.")
    if "제목_전처리" not in df.columns and "제목" not in df.columns:
        print("⚠️ '제목_전처리' 또는 '제목' 컬럼이 없어 텍스트 길이 관련 그래프는 일부 스킵될 수 있음")

    SENT_COL = find_sentiment_col(df)
    print("감성 컬럼:", SENT_COL)

    # sentiment 라벨 통일 (-1/1 또는 0/1 가능)
    s = df[SENT_COL].copy()
    # 0/1이면 -1/1로 매핑(0->-1, 1->1)
    uniq = set(pd.unique(s.dropna()))
    if uniq.issubset({0, 1}):
        s = s.map({0: -1, 1: 1})
    df["_sent"] = s

    # ==========================
    # 1) 전체 감성 분포
    # ==========================
    value_counts_plot(
        df["_sent"].map({-1: "부정(-1)", 1: "긍정(1)"}).fillna("기타/결측"),
        "전체 감성 분포",
        "01_overall_sentiment_distribution.png",
        xlabel="sentiment"
    )

    # ==========================
    # 2) 종목별 댓글 수 분포 (TOP 15)
    # ==========================
    cnt_by_ticker = df["종목명"].value_counts().head(15)
    plt.figure(figsize=(10,6))
    plt.bar(cnt_by_ticker.index.astype(str), cnt_by_ticker.values)
    plt.title("종목별 댓글 수 TOP 15")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("댓글 수")
    save_fig("02_comment_count_top15.png")

    # ==========================
    # 3) 종목별 감성 스코어 계산 (긍정%-부정%)
    # ==========================
    g = df.groupby("종목명")["_sent"]
    summary = pd.DataFrame({
        "전체댓글수": g.size(),
        "긍정수": g.apply(lambda x: (x == 1).sum()),
        "부정수": g.apply(lambda x: (x == -1).sum()),
    }).reset_index()

    summary["긍정비율(%)"] = (summary["긍정수"] / summary["전체댓글수"] * 100).round(2)
    summary["부정비율(%)"] = (summary["부정수"] / summary["전체댓글수"] * 100).round(2)
    summary["감성스코어"] = (summary["긍정비율(%)"] - summary["부정비율(%)"]).round(2)

    # 저장(보고서 표로도 쓰기 좋음)
    out_csv = os.path.join(os.path.dirname(OUT_DIR), "sentiment_by_ticker_from_viz.csv")
    summary.sort_values("감성스코어", ascending=False).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("✅ 종목 요약 CSV 저장:", out_csv)

    # ==========================
    # 4) 감성 스코어 TOP / BOTTOM 10
    # ==========================
    top10 = summary.sort_values("감성스코어", ascending=False).head(10)
    bot10 = summary.sort_values("감성스코어", ascending=True).head(10)

    plt.figure(figsize=(10,6))
    plt.bar(top10["종목명"].astype(str), top10["감성스코어"].values)
    plt.title("감성 스코어 TOP 10 (긍정 우세)")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("감성스코어(긍정%-부정%)")
    save_fig("03_sentiment_score_top10.png")

    plt.figure(figsize=(10,6))
    plt.bar(bot10["종목명"].astype(str), bot10["감성스코어"].values)
    plt.title("감성 스코어 BOTTOM 10 (부정 우세)")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("감성스코어(긍정%-부정%)")
    save_fig("04_sentiment_score_bottom10.png")

    # ==========================
    # 5) 종목별 긍/부정 비율 비교 (TOP 10만)
    # ==========================
    t = top10.copy()
    x = np.arange(len(t))
    width = 0.35

    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, t["긍정비율(%)"], width, label="긍정비율(%)")
    plt.bar(x + width/2, t["부정비율(%)"], width, label="부정비율(%)")
    plt.title("TOP10 종목 긍/부정 비율 비교")
    plt.xticks(x, t["종목명"].astype(str), rotation=45, ha="right")
    plt.ylabel("비율(%)")
    plt.legend()
    save_fig("05_top10_pos_neg_ratio.png")

    # ==========================
    # 6) 텍스트 길이 분포(전처리 텍스트 기준) + 감성별 비교
    # ==========================
    text_col = "제목_전처리" if "제목_전처리" in df.columns else ("제목" if "제목" in df.columns else None)
    if text_col:
        df["_len"] = df[text_col].astype(str).apply(len)

        # 전체 길이 히스토그램
        plt.figure(figsize=(10,6))
        plt.hist(df["_len"].values, bins=30)
        plt.title(f"텍스트 길이 분포 ({text_col})")
        plt.xlabel("length")
        plt.ylabel("count")
        save_fig("06_text_length_hist.png")

        # 감성별 길이 비교(박스플롯)
        pos_len = df[df["_sent"] == 1]["_len"].values
        neg_len = df[df["_sent"] == -1]["_len"].values
        plt.figure(figsize=(8,6))
        plt.boxplot([neg_len, pos_len], labels=["부정(-1)", "긍정(1)"], showfliers=False)
        plt.title("감성별 텍스트 길이 비교(박스플롯)")
        plt.ylabel("length")
        save_fig("07_text_length_by_sentiment_box.png")

    # ==========================
    # 7) 날짜 컬럼이 있으면 시계열(일자별 감성 비율)
    # ==========================
    # 흔한 날짜 컬럼 후보들
    date_candidates = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "날짜", "작성일"])]
    date_col = date_candidates[0] if date_candidates else None

    if date_col:
        tmp = df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col])
        if len(tmp) > 0:
            tmp["day"] = tmp[date_col].dt.date
            day_g = tmp.groupby("day")["_sent"]
            day_summary = pd.DataFrame({
                "total": day_g.size(),
                "pos_ratio": (day_g.apply(lambda x: (x == 1).mean()) * 100),
                "neg_ratio": (day_g.apply(lambda x: (x == -1).mean()) * 100),
            }).reset_index()

            plt.figure(figsize=(12,6))
            plt.plot(day_summary["day"], day_summary["pos_ratio"], marker="o")
            plt.title("일자별 긍정 비율(%)")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("pos_ratio(%)")
            save_fig("08_daily_positive_ratio.png")

            plt.figure(figsize=(12,6))
            plt.plot(day_summary["day"], day_summary["neg_ratio"], marker="o")
            plt.title("일자별 부정 비율(%)")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("neg_ratio(%)")
            save_fig("09_daily_negative_ratio.png")
        else:
            print("⚠️ 날짜 컬럼은 있으나 파싱 실패/결측이 많아 시계열 스킵:", date_col)
    else:
        print("ℹ️ 날짜 컬럼을 찾지 못해 시계열 그래프는 스킵")

    # ==========================
    # 8) 종목별 감성스코어 vs 댓글수 산점도(전체)
    # ==========================
    plt.figure(figsize=(10,6))
    plt.scatter(summary["전체댓글수"], summary["감성스코어"])
    plt.title("종목별 댓글 수 vs 감성 스코어")
    plt.xlabel("전체댓글수")
    plt.ylabel("감성스코어")
    save_fig("10_scatter_count_vs_score.png")

    print("\n🎉 시각화 생성 완료! →", OUT_DIR)

if __name__ == "__main__":
    main()
