"""
영어 감정 데이터셋을 한국어로 번역하는 스크립트
GPT API를 사용하여 번역 수행
"""
import os
import json
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from pathlib import Path

def translate_to_korean(text: str, client: OpenAI) -> str:
    """
    GPT API를 사용하여 영어 텍스트를 한국어로 번역
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 전문 번역가입니다. 영어 텍스트를 자연스러운 한국어로 번역하세요. 감정과 뉘앙스를 최대한 보존하세요."
                },
                {
                    "role": "user",
                    "content": f"다음 영어 문장을 한국어로 번역하세요. 번역된 문장만 출력하세요:\n\n{text}"
                }
            ],
            temperature=0.3,
            max_completion_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"번역 실패: {text[:50]}... - 오류: {e}")
        return text  # 실패시 원문 반환


def translate_dataset(split="train", max_samples=None, output_dir="./korean_emotion_data"):
    """
    데이터셋을 번역하여 저장

    Args:
        split: "train" 또는 "test"
        max_samples: 번역할 최대 샘플 수 (None이면 전체)
        output_dir: 저장할 디렉토리
    """
    # OpenAI 클라이언트 초기화
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수를 설정해주세요.")

    client = OpenAI(api_key=api_key)

    # 데이터셋 로드
    print(f"[INFO] {split} 데이터셋 로딩 중...")
    dataset = load_dataset("dair-ai/emotion", "split")
    data = dataset[split]

    # 샘플 수 제한
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    print(f"[INFO] 총 {len(data)}개 샘플 번역 시작")

    # 번역 수행
    translated_data = []
    for i, item in enumerate(tqdm(data, desc=f"Translating {split}")):
        korean_text = translate_to_korean(item["text"], client)

        translated_data.append({
            "text": korean_text,
            "text_en": item["text"],  # 원문 보존
            "label": item["label"],
        })

        # 10개마다 중간 저장
        if (i + 1) % 10 == 0:
            save_intermediate(translated_data, split, output_dir)

    # 최종 저장
    save_final(translated_data, split, output_dir)
    print(f"[INFO] 번역 완료! {len(translated_data)}개 저장됨")


def save_intermediate(data, split, output_dir):
    """중간 저장 (백업용)"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    temp_file = output_path / f"{split}_temp.json"
    with temp_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_final(data, split, output_dir):
    """최종 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # JSON 형식으로 저장
    json_file = output_path / f"{split}.json"
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[저장] {json_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("데이터셋 한국어 번역")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "validation"],
        help="번역할 데이터 분할"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="번역할 최대 샘플 수 (테스트용)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./korean_emotion_data",
        help="저장 디렉토리"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("데이터셋 한국어 번역 시작")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Max samples: {args.max_samples or '전체'}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    translate_dataset(
        split=args.split,
        max_samples=args.max_samples,
        output_dir=args.output_dir
    )
