import pandas as pd
import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from tqdm import tqdm

MODEL_DIR = "../model/koelectra_binary_sentiment"
INPUT_PATH = "../data/naver_board_kospi100_cleaned_final.csv"
OUTPUT_PATH = "../data/naver_board_kospi100_with_sentiment.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

def load_model():
    tokenizer = ElectraTokenizer.from_pretrained(MODEL_DIR)
    model = ElectraForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    model.to(device)
    model.eval()
    return tokenizer, model

def predict(texts, tokenizer, model):
    results = []
    for t in tqdm(texts):
        inputs = tokenizer(
            t,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=1).item()

        # 0 → 부정(-1), 1 → 긍정(+1)
        label = 1 if pred == 1 else -1
        results.append(label)

    return results

def main():
    print("데이터 로드 중...")
    df = pd.read_csv(INPUT_PATH, encoding="utf-8")
    texts = df["제목_전처리"].astype(str).tolist()

    tokenizer, model = load_model()

    print("전체 데이터 감성 분석 중...")
    df["sentiment_binary"] = predict(texts, tokenizer, model)

    print("\n저장합니다 →", OUTPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("\n완료 🎉")

if __name__ == "__main__":
    main()
