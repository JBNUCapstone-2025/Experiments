import random
from typing import List,Dict
from datasets import Dataset

#sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).
EMOTION_EN = ["sadness","joy","love","anger","fear","surprise"]

def build_few_shot_examples(dataset, k_per_label = 2, seed = 42):
    """
    각 감정 레이블당 k_per_label개의 예시를 생성
    예) k_per_label=1 -> 총 6개 예시 (각 감정당 1개씩)
        k_per_label=2 -> 총 12개 예시 (각 감정당 2개씩)
    """
    random.seed(seed)
    buckets = {i: [] for i in range(len(EMOTION_EN))}  # 0~5까지 초기화

    for ex in dataset:
        label_id = ex["label"]
        buckets[label_id].append(ex)

    examples = []
    # 각 레이블별로 순서대로 처리
    for label_id in range(len(EMOTION_EN)):
        samples = buckets[label_id]
        chosen = random.sample(samples, min(k_per_label, len(samples)))
        for ex in chosen:
            emo_en = EMOTION_EN[label_id]
            examples.append({
                "text": ex["text"],
                "emotion_en": emo_en,
            })

    return examples

def build_few_shot_prompt(few_shot_examples, user_input):

    label_list_en = ", ".join(EMOTION_EN)

    header = (
        '''
        You are an assistant that classifies the writer's primary emotion.

        RULES:
        - Focus only on the writer's emotional state.
        - Ignore topics, opinions, moral/political content.
        - Determine the dominant emotion based on tone, wording, and implied sentiment.

        Valid emotions (choose exactly one):
        sadness, joy, love, anger, fear, surprise

        Definitions:
        - sadness: loss, disappointment, loneliness, hurt
        - joy: happiness, interest, excitement, positivity
        - love: affection, warmth, closeness, caring
        - anger: frustration, irritation, resentment
        - fear: anxiety, worry, tension
        - surprise: shock, confusion, unexpectedness

        OUTPUT:
        Return only one emotion word. No explanation.
        '''
    )

    examples_str = ""
    if few_shot_examples:
        examples_str += "\n[Examples]\n\n"
        for ex in few_shot_examples:
            examples_str += (
                f'Sentence: {ex["text"]}\n'
                f'Emotion: {ex["emotion_en"]}\n\n'
            )

    query = (
        "\n[Classification Target]\n\n"
        f"Sentence: {user_input}\n"
        "Emotion: "
    )

    return header + examples_str + query


def extract_emotion_en_from_output(output_text):
    text = output_text.strip().lower()

    for emo_en in EMOTION_EN:
        if emo_en in text:
            return emo_en

    return text

def label_id_to_emotion(label_id):
    return EMOTION_EN[label_id]