# gpt-4o-mini
import os
from openai import OpenAI

# MODEL_NAME = "ft:gpt-4o-mini-2024-07-18:personal:emotion:CktCR1Hx"
MODEL_NAME = "gpt-4o-mini"

_client = None


def _init_client():
    global _client
    if _client is not None:
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY 환경 변수를 환경변수로 설정해주세요.")
    _client = OpenAI(api_key=api_key)


def generate(prompt: str, max_new_tokens: int = 32, gpu_id=None) -> str:
    """
    OpenAI GPT 모델로 텍스트 생성
    gpu_id는 API 모델에서는 무시됨
    """
    _init_client()

    resp = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_new_tokens,
    )
    return resp.choices[0].message.content.strip()
