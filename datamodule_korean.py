import random
import json
from pathlib import Path
from datasets import Dataset

class KoreanEmotionDataset:
    """
    한국어로 번역된 감정 데이터셋
    """
    def __init__(self, data_dir, seed, split):
        """
        Args:
            data_dir: 한국어 데이터가 저장된 디렉토리 (korean_emotion_data/)
            seed: 랜덤 시드
            split: "train" 또는 "test"
        """
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.split = split

        # JSON 파일 로드
        json_file = self.data_dir / f"{split}.json"
        if not json_file.exists():
            raise FileNotFoundError(
                f"한국어 데이터셋이 없습니다: {json_file}\n"
                f"먼저 translate_dataset.py를 실행하여 데이터를 번역하세요."
            )

        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # HuggingFace Dataset으로 변환
        self.dataset = Dataset.from_list(data)

        # 레이블 이름
        self.label_name = ["sadness", "joy", "love", "anger", "fear", "surprise"]

        # 셔플
        random.seed(seed)
        idxs = list(range(len(self.dataset)))
        random.shuffle(idxs)
        self.dataset = self.dataset.select(idxs)

        print(f"[INFO] 한국어 데이터셋 로드 완료: {len(self.dataset)}개 샘플 ({split})")

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        return {
            "text": ex["text"],  # 한국어 텍스트
            "text_en": ex.get("text_en", ""),  # 영어 원문 (있으면)
            "label_id": ex["label"],
            "label": self.label_name[ex["label"]],
        }

    def __len__(self):
        return len(self.dataset)
