import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_PATH = "../data/naver_board_kospi100_with_sentiment.csv"
OUT_DIR = "../results/figures_clean"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["axes.unicode_minus"] = False
try:
    plt.rcParams["font.family"] = "Malgun Gothic"
except:
    pass

def save_fig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    print("✅ 저장:", path)
    plt.close()

def safe_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", str(name))
    return name[:80]

def find_sentiment_col(df):
    cand = [c for c in df.columns if "sentiment" in c.lower()]
    if cand:
        return cand[-1]
    if "label" in df.columns:
        return "label"
    raise KeyError("sentiment 컬럼을 찾을 수 없음")

def hbar_top(df, col_name, value_name, title, filename, topn=10):
    sub = df.head(topn).iloc[::-1]  # 아래에서 위로 보기 좋게
    plt.figure(figsize=(10, 6))
    plt.barh(sub[col_name].astype(str), sub[value_name].values)
    plt.title(title)
    plt.xlabel(value_name)
    # 값 라벨
    mx = sub[value_name].max() if len(sub) else 0
    for y, v in enumerate(sub[value_name].values):
        plt.text(v + (mx * 0.01 if mx else 0.5), y, f"{v:.2f}" if isinstance(v, float) else str(v),
                 va="center")
    save_fig(filename)

def main():
    df = pd.read_csv(INPUT_PATH, encoding="utf-8")
    print("행:", len(df), "컬럼:", df.columns.tolist())

    if "종목명" not in df.columns:
        raise KeyError("'종목명' 컬럼이 없습니다.")

    sent_col = find_sentiment_col(df)
    s = df[sent_col].copy()

    # 0/1이면 -1/1로 통일
    uniq = set(pd.unique(s.dropna()))
    if uniq.issubset({0, 1}):
        s = s.map({0: -1, 1: 1})
    df["_sent"] = s

    # ==========================
    # (1) 전체 감성 분포 (퍼센트 라벨)
    # ==========================
    vc = df["_sent"].value_counts().reindex([-1, 1]).fillna(0).astype(int)
    labels = ["부정(-1)", "긍정(1)"]
    vals = [vc.get(-1, 0), vc.get(1, 0)]
    total = sum(vals) if sum(vals) else 1

    plt.figure(figsize=(7,5))
    bars = plt.bar(labels, vals)
    plt.title("전체 감성 분포")
    for i, v in enumerate(vals):
        plt.text(i, v + max(vals)*0.02, f"{v} ({v/total*100:.1f}%)", ha="center")
    save_fig("01_overall_sentiment.png")

    # ==========================
    # (2) 종목별 댓글 수 TOP15 (가로 막대)
    # ==========================
    cnt = df["종목명"].value_counts().reset_index()
    cnt.columns = ["종목명", "댓글수"]
    cnt_top = cnt.head(15)
    hbar_top(cnt_top, "종목명", "댓글수", "종목별 댓글 수 TOP 15", "02_count_top15.png", topn=15)

    # ==========================
    # (3) 종목별 감성 스코어 계산
    # ==========================
    g = df.groupby("종목명")["_sent"]
    summary = pd.DataFrame({
        "종목명": g.size().index,
        "전체댓글수": g.size().values,
        "긍정수": g.apply(lambda x: (x==1).sum()).values,
        "부정수": g.apply(lambda x: (x==-1).sum()).values,
    })
    summary["긍정비율(%)"] = (summary["긍정수"]/summary["전체댓글수"]*100).round(2)
    summary["부정비율(%)"] = (summary["부정수"]/summary["전체댓글수"]*100).round(2)
    summary["감성스코어"] = (summary["긍정비율(%)"] - summary["부정비율(%)"]).round(2)

    # 결과표 저장(보고서 표/부록용)
    summary.sort_values("감성스코어", ascending=False).to_csv(
        "../results/sentiment_by_ticker_clean.csv", index=False, encoding="utf-8-sig"
    )

    # ==========================
    # (4) 감성 스코어 TOP/BOTTOM 10 (가로 막대)
    # ==========================
    top10 = summary.sort_values("감성스코어", ascending=False).head(10)
    bot10 = summary.sort_values("감성스코어", ascending=True).head(10)

    hbar_top(top10, "종목명", "감성스코어", "감성 스코어 TOP 10 (긍정 우세)", "03_score_top10.png", topn=10)
    hbar_top(bot10, "종목명", "감성스코어", "감성 스코어 BOTTOM 10 (부정 우세)", "04_score_bottom10.png", topn=10)

    # ==========================
    # (5) TOP10 종목 긍/부정 비율 비교(한 장)
    # ==========================
    t = top10.sort_values("감성스코어", ascending=True)  # 보기 좋게
    y = np.arange(len(t))
    plt.figure(figsize=(10,6))
    plt.barh(y - 0.2, t["긍정비율(%)"], height=0.4, label="긍정비율(%)")
    plt.barh(y + 0.2, t["부정비율(%)"], height=0.4, label="부정비율(%)")
    plt.yticks(y, t["종목명"].astype(str))
    plt.title("TOP10 종목 긍/부정 비율 비교")
    plt.xlabel("비율(%)")
    plt.legend()
    save_fig("05_top10_pos_neg_ratio.png")

    print("\n🎉 핵심 시각화만 깔끔하게 생성 완료 →", OUT_DIR)

if __name__ == "__main__":
    main()
