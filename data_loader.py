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
        "다음 문장을 보고 가장 잘 맞는 핵심 감정을 하나만 선택해주세요.\n"
        f"가능한 감정 레이블은 다음 여섯 가지입니다:\n{label_list_ko}\n\n"
        "설명 없이 감정 단어만 한국어로 한 단어로 답변해주세요.\n"
    )
    # header = (
    #     "당신은 텍스트에 담긴 화자의 미묘한 심리를 파악하는 'AI 감정 분석 전문가'입니다.\n"
    #     "주어진 문장을 깊이 있게 읽고, 화자가 느끼는 가장 지배적인 감정을 분석하세요.\n\n"
    #     f"선택 가능한 감정 레이블:\n{label_list_ko}\n\n"
    #     "규칙:\n"
    #     "1. 문장의 표면적인 단어보다 문맥과 숨겨진 의도를 파악하세요.\n"
    #     "2. 반드시 위 목록에 있는 단어 중 하나만 선택하세요.\n"
    #     "3. 설명이나 부가적인 말 없이 오직 '감정 단어' 하나만 출력하세요."
    # )

    # header = (
    #     "다음 문장의 감정을 분석하여 가장 적절한 레이블을 선택해야 합니다.\n"
    #     f"가능한 감정 레이블: {label_list_ko}\n\n"
    #     "다음 단계를 거쳐 답변하세요:\n"
    #     "1. [Reasoning]: 문장의 상황과 화자의 어조를 분석하여 이유를 한 문장으로 서술.\n"
    #     "2. [Label]: 분석에 근거하여 가장 적절한 감정 레이블 선택.\n\n"
    #     "최종 답변은 반드시 다음과 같은 JSON 형식으로만 출력하세요:\n"
    #     '{"reasoning": "...", "label": "..."}'
    # )

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
        "감정 :"
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