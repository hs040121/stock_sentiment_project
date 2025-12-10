import pandas as pd
import os

# ==========================
# 설정
# ==========================
INPUT_PATH = "../data/naver_board_kospi100_with_sentiment.csv"
OUTPUT_DIR = "../results"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sentiment_by_ticker.csv")

# 결과 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# 메인 함수
# ==========================
def main():
    print("데이터 로드 중...")
    df = pd.read_csv(INPUT_PATH, encoding="utf-8")

    # 감성 컬럼 자동 탐지
    sentiment_col = None
    for c in df.columns:
        if "sentiment" in c.lower():
            sentiment_col = c
            break

    if sentiment_col is None:
        raise KeyError("❌ 감성 컬럼을 찾을 수 없음 (예: sentiment_binary)")

    print(f"감성 컬럼 사용: {sentiment_col}")

    # ==========================
    # 종목별 감성 분석
    # ==========================
    grouped = df.groupby("종목명")[sentiment_col]

    results = []
    for stock, series in grouped:
        total = len(series)
        pos = (series == 1).sum()
        neg = (series == -1).sum()

        pos_ratio = round(pos / total * 100, 2)
        neg_ratio = round(neg / total * 100, 2)
        sentiment_score = pos_ratio - neg_ratio

        results.append({
            "종목명": stock,
            "전체댓글수": total,
            "긍정수": pos,
            "부정수": neg,
            "긍정비율(%)": pos_ratio,
            "부정비율(%)": neg_ratio,
            "감성스코어": sentiment_score
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="감성스코어", ascending=False)

    # 저장
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n📁 파일 저장 완료 →", OUTPUT_PATH)

    # ==========================
    # 상위 / 하위 종목 출력
    # ==========================
    print("\n📌 긍정 높은 종목 TOP 10")
    print(result_df.head(10).to_string(index=False))

    print("\n📌 부정 높은 종목 TOP 10")
    print(result_df.tail(10).sort_values(by="감성스코어").to_string(index=False))

    print("\n🎉 종목별 감성 분석 완료!")


if __name__ == "__main__":
    main()
