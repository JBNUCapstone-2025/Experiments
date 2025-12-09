from datasets import load_dataset
import json

DATA_PATH = "dair-ai/emotion"

# 1. 데이터 로드
ds = load_dataset(DATA_PATH)

# 2. 감정 라벨 매핑
label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# 3. jsonl 파일 생성
output_file = "emotion_train.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for record in ds["train"]:
        text = record["text"]
        label = label_map[record["label"]]

        item = {
            "messages": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": label}
            ]
        }

        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"완료! jsonl 파일 생성: {output_file}")
