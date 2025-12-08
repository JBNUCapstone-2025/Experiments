import random
from typing import List,Dict
from datasets import Dataset

#sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).
EMOTION_EN = ["sadness","joy","love","anger","fear","surprise"]
EMOTION_KOR = ["슬픔","기쁨","설렘","분노","불안","놀람"]

EN_TO_KO = {
    "sadness": "슬픔",
    "joy": "기쁨",
    "love": "설렘",
    "anger": "분노",
    "fear": "불안",
    "surprise": "놀람",
}

KO_TO_EN = {v: k for k, v in EN_TO_KO.items()}

def build_few_shot_examples(dataset, k_per_label = 2, seed = 42):
    random.seed(seed)
    buckets = {}

    for ex in dataset:
        label_id = ex["label"]
        buckets.setdefault(label_id, []).append(ex)
    
    examples = []
    for label_id, samples in buckets.items():
        chosen = random.sample(samples, min(k_per_label, len(samples)))
        for ex in chosen:
            emo_en = EMOTION_EN[label_id]
            emo_ko = EN_TO_KO[emo_en]
            examples.append({
                "text": ex["text"],
                "emotion_ko" : emo_ko,
                "emotion_en" : emo_en,
            })

    return examples

def build_few_shot_prompt(few_shot_examples, user_input):

    label_list_ko = ",".join(EMOTION_KOR)

    header = (
        "다음 문장의 감정을 분석하고, 아래 6가지 감정 중 하나만 선택하세요.\n\n"
        f"감정 목록: {label_list_ko}\n\n"
        "중요: 반드시 위 목록의 단어 중 정확히 하나만 출력하세요. 설명, 번호, 기호 등은 출력하지 마세요.\n"
    )

    examples_str = ""
    if few_shot_examples:
        examples_str += "\n[예시]\n\n"
        for ex in few_shot_examples:
            examples_str += (
                f'문장 : {ex["text"]}\n'
                f'감정 : {ex["emotion_ko"]}\n\n'
            )

    query = (
        "\n[분류 대상]\n\n"
        f'문장 : {user_input}\n'
        "감정 : "
    )

    return header + examples_str + query

def extrat_emotion_ko_from_output(output_text):
    text = output_text.strip()

    for emo_ko in EMOTION_KOR:
        if emo_ko in text:
            return emo_ko

    lower = text.lower()
    for emo_en, emo_ko in EN_TO_KO.items():
        if emo_en in lower:
            return emo_ko
    
    return text

def ko_to_en_motion(emo_ko):
    emo_ko = emo_ko.strip()
    return KO_TO_EN.get(emo_ko,"unknown")

def en_id_to_ko(label_id):
    emo_en = EMOTION_EN[label_id]
    return EN_TO_KO[emo_en]