import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# === 한글 폰트 설정 (Windows용) ===
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)   # 음수 깨짐 방지



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 파일 로드 =====
df = pd.read_csv("../data/naver_board_kospi100_with_sentiment.csv")
print("총 행:", len(df))

# ===== 감성 컬럼 자동 탐지 =====
sent_cols = [c for c in df.columns if "sentiment" in c.lower()]

if len(sent_cols) == 0:
    raise ValueError("❌ sentiment 관련 컬럼을 찾을 수 없습니다.")
else:
    print("감성 컬럼 자동 탐지됨:", sent_cols)

# 가장 마지막(가장 최신) sentiment 컬럼 사용
SENT_COL = sent_cols[-1]
print("👉 사용 감성 컬럼:", SENT_COL)

# ===== 감성 분포 =====
sent_count = df[SENT_COL].value_counts().sort_index()

labels = ["부정(0)", "긍정(1)"]
colors = ["#FF637D", "#3FA7D6"]

plt.figure(figsize=(10,6))
bars = plt.bar(labels, sent_count, color=colors, edgecolor="black")

for i, val in enumerate(sent_count):
    plt.text(i, val + max(sent_count)*0.02,
             f"{val} ({val/len(df)*100:.1f}%)",
             ha="center", fontsize=13, fontweight="bold")

plt.title("전체 감성 분포")
plt.tight_layout()
plt.show()

print("\n🎉 감성 EDA 완료!")
