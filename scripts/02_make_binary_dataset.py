import pandas as pd
import numpy as np
import re
from sklearn.utils import shuffle

# =======================================
# 파일 경로 설정
# =======================================
INPUT_PATH = "./data/raw/naver_board_kospi100_cleaned.csv"
FULL_OUTPUT = "../data/naver_board_kospi100_labeled_full_17k.csv"
BALANCED_OUTPUT = "../data/balanced_2000_binary_dataset.csv"

TEXT_COL = "제목_전처리"

# =======================================
# 감성 키워드 사전
# =======================================
POS_STRONG = [
    "상한가", "급등", "폭등", "반등", "대박", "호재", "수익", "흑자",
    "기대", "좋다", "좋네", "가즈아", "가자", "우상향", "상승장",
    "불장", "축하", "축하합니다", "고맙다", "고마워", "신고가",
]

NEG_STRONG = [
    "폭락", "급락", "하락", "추락", "손실", "손절", "물렸다",
    "망함", "망했다", "휴지조각", "쓰레기", "개잡주", "사기",
    "공매도", "악재", "지옥", "멘붕", "최악", "양아치",
    "상폐", "상장폐지", "국장쓰레기", "거지같", "죽었다",
]

POS_WEAK = [
    "ㅋㅋ", "ㅎㅎ", "^^", "이득", "이득봤다", "기분좋", "좋구만", "오늘은웃는다"
]

NEG_WEAK = [
    "왜이래", "뭐하냐", "뭐냐", "어이없", "미친", "개판", "답이없",
    "망한거", "징그럽", "나라망했", "환장", "욕나온다"
]

# =======================================
# 감성 라벨 함수 (-1 부정 / 1 긍정)
# =======================================
def classify_sentiment(text: str) -> int:
    if not isinstance(text, str):
        return -1

    t = text.replace(" ", "")

    pos_hits = sum(1 for w in POS_STRONG if w in t)
    neg_hits = sum(1 for w in NEG_STRONG if w in t)

    # 강한 단어 우선 판정
    if pos_hits > 0 and neg_hits == 0:
        return 1
    if neg_hits > 0 and pos_hits == 0:
        return -1
    if pos_hits > 0 and neg_hits > 0:
        return -1  # 섞이면 부정으로

    # 약한 단어 판정
    if any(w in t for w in POS_WEAK):
        return 1
    if any(w in t for w in NEG_WEAK):
        return -1

    # 아무것도 안 맞으면 중립 제거 → 기본 부정
    return -1

# =======================================
# 메인 로직
# =======================================
def main():
    print("데이터 로드 중...")
    df = pd.read_csv(INPUT_PATH, encoding="utf-8")

    if TEXT_COL not in df.columns:
        raise KeyError(f"❌ '{TEXT_COL}' 컬럼이 CSV에 없음!")

    print("전체 텍스트 수:", len(df))

    # -----------------------------
    # 1) 전체 17k 라벨링
    # -----------------------------
    print("\n전체 감성 라벨링 중...")
    df["label"] = df[TEXT_COL].apply(classify_sentiment)

    print("\n라벨 분포:")
    print(df["label"].value_counts())

    df.to_csv(FULL_OUTPUT, index=False, encoding="utf-8-sig")
    print("\n✔ 저장 완료 →", FULL_OUTPUT)

    # -----------------------------
    # 2) Balanced 2000 생성
    # -----------------------------
    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == -1]

    n = min(len(pos_df), len(neg_df), 1000)

    balanced = pd.concat([
        pos_df.sample(n=n, random_state=42),
        neg_df.sample(n=n, random_state=42)
    ], axis=0)

    balanced = shuffle(balanced, random_state=42)
    balanced = balanced[[TEXT_COL, "label"]]

    balanced.to_csv(BALANCED_OUTPUT, index=False, encoding="utf-8-sig")

    print("\nBalanced Dataset 저장 완료 →", BALANCED_OUTPUT)
    print(f"(긍정 {n}개, 부정 {n}개, 총 {len(balanced)}개)")

    print("\n🎉 완료!")

if __name__ == "__main__":
    main()
