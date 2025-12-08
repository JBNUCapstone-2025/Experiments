# Qwen/Qwen2-7B-Instruct

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"

tokenizer = None
model = None
device = None

def init_model(gpu_id=None):
    global tokenizer, model, device

    if model is not None:
        return

    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[QWEN] Loading model on {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype = torch.float16 if "cuda" in device else torch.float32,
    )
    model.to(device)
    model.eval()

def generate(prompt, max_new_tokens, gpu_id=None):

    init_model(gpu_id)

    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        truncation = True,
        max_length = 2048,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample = False,
            pad_token_id = tokenizer.eos_token_id,
            eos_token_id = tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens = True).strip()

    # 불필요한 부분 제거 (첫 줄바꿈 이전까지만 사용)
    if '\n' in text:
        text = text.split('\n')[0].strip()

    return text