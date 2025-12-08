# meta-llama/Llama-3.1-8B-Instruct
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

_tokenizer = None
_model = None
_device = None


def _init_model(gpu_id=None):
    global _tokenizer, _model, _device
    if _model is not None:
        return

    if gpu_id is not None:
        _device = f"cuda:{gpu_id}"
    else:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LLaMA] Loading model: {MODEL_NAME} on {_device}")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if "cuda" in _device else torch.float32,
    )
    _model.to(_device)
    _model.eval()


def generate(prompt: str, max_new_tokens: int = 32, gpu_id=None) -> str:
    """
    LLaMA-3 Instruct 모델로 텍스트 생성
    (chat template 사용 버전)
    """
    _init_model(gpu_id)

    # 1) chat 메시지 포맷으로 감싸기
    messages = [
        {"role": "user", "content": prompt}
    ]

    # 2) LLaMA 전용 chat template 적용
    input_ids = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(_device)

    # 3) 생성
    with torch.no_grad():
        outputs = _model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    # 4) 프롬프트 이후의 토큰만 잘라서 디코딩
    gen_ids = outputs[0][input_ids.shape[1]:]
    text = _tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # 5) 첫 줄만 사용 (감정 레이블 한 단어만 뽑고 싶은 경우)
    if "\n" in text:
        text = text.split("\n")[0].strip()

    return text
