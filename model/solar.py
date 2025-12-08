# upstage/SOLAR-10.7B-Instruct-v1.0
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "upstage/SOLAR-10.7B-Instruct-v1.0"

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
    print(f"[SOLAR] Loading model: {MODEL_NAME} on {_device}")

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
    SOLAR 모델로 텍스트 생성
    """
    _init_model(gpu_id)

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(_device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = _tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()
