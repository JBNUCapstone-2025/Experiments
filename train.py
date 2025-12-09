import argparse
from pathlib import Path
import torch.multiprocessing as mp
from multiprocessing import Manager
import json

from datamodule import EmotionDataset
from data_loader import (
    build_few_shot_examples,
    build_few_shot_prompt,
    extract_emotion_en_from_output,
    label_id_to_emotion,
    EMOTION_EN,
)

import model.qwen as qwen
import model.llama as llama
import model.ministral as ministral
import model.solar as solar
import model.gpt as gpt

DATA_PATH = "dair-ai/emotion"

def get_model_generate(model_name):
    model_name = model_name.lower()

    if model_name == "qwen":
        return qwen.generate
    elif model_name == "llama":
        return llama.generate
    elif model_name == "ministral":
        return ministral.generate
    elif model_name == "solar":
        return solar.generate
    elif model_name == "gpt":
        return gpt.generate
    else:
        raise ValueError(f"Unkown model name!")

def parse_args():
    parser = argparse.ArgumentParser("Few-shot Emotion Classificaiton")
    parser.add_argument(
        "--model_name",
        type = str,
        required = True,
        help = "qwen , llama, ministrial, solar, gpt"
    )
    parser.add_argument(
        "--shots",
        type = int,
        default = 2,
        help = "few-shot 수"
    )
    parser.add_argument(
        "--samples",
        type = int,
        default = 200,
        help = "테스트에 사용할 최대 샘플 수"
    )
    parser.add_argument(
        "--gpus",
        type = str,
        default = "0,1,2",
        help = "사용할 GPU IDs (콤마로 구분, 예: 0,1,2)"
    )
    parser.add_argument(
        "--result_name",
        type = str,
        default = None,
        help = "결과 파일명 (기본값: 모델명_result.txt)"
    )
    parser.add_argument(
        "--save_io",
        action = "store_true",
        help = "입력/출력을 JSON 파일로 저장"
    )
    parser.add_argument(
        "--io_output_file",
        type = str,
        default = None,
        help = "입출력 저장 파일명 (기본값: {model_name}_io.json)"
    )
    return parser.parse_args()

def worker_process(gpu_id, model_name, few_shot_examples, test_samples, results_dict, worker_id, save_io=False, io_list=None):
    """
    각 GPU에서 실행될 워커 프로세스
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[Worker {worker_id}] GPU {gpu_id}에서 {len(test_samples)}개 샘플 처리 시작")

    generate = get_model_generate(model_name)

    correct = 0
    total = 0

    for i, sample in enumerate(test_samples):
        text = sample["text"]
        label_id = sample["label_id"]

        gold_en = label_id_to_emotion(label_id)

        prompt = build_few_shot_prompt(few_shot_examples, text)
        output = generate(prompt, max_new_tokens=8, gpu_id=0)  # CUDA_VISIBLE_DEVICES로 인해 항상 0

        pred_en = extract_emotion_en_from_output(output)

        total += 1
        is_correct = (pred_en == gold_en)
        if is_correct:
            correct += 1

        # 입출력 저장
        if save_io and io_list is not None:
            io_record = {
                "sample_index": i,
                "input_text": text,
                "prompt": prompt,
                "model_output": output,
                "predicted_emotion": pred_en,
                "gold_emotion": gold_en,
                "gold_label_id": label_id,
                "is_correct": is_correct
            }
            io_list.append(io_record)

        if (i + 1) % 10 == 0:
            print(f"[Worker {worker_id}] 진행: {i+1}/{len(test_samples)}")

    results_dict[worker_id] = {"correct": correct, "total": total}
    print(f"[Worker {worker_id}] 완료: {correct}/{total}")
    
def main():
    mp.set_start_method('spawn', force=True)

    args = parse_args()

    model_name = args.model_name.lower()
    K_PER_LABEL = args.shots
    MAX_SAMPLES = args.samples
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]

    print(f"[INFO] 모델 선택 : {model_name}")
    print(f"[INFO] Few-shot per label : {K_PER_LABEL}")
    print(f"[INFO] 사용할 GPU : {gpu_ids}")

    # 데이터셋 로드
    val_ds = EmotionDataset(DATA_PATH, seed=42, split="validation")
    test_ds = EmotionDataset(DATA_PATH, seed=42, split="test")

    # Few-shot examples 생성 (K_PER_LABEL > 0일 때만)
    if K_PER_LABEL > 0:
        few_shot_examples = build_few_shot_examples(val_ds.dataset, K_PER_LABEL)
    else:
        few_shot_examples = []

    # 테스트 샘플 준비
    num_samples = min(MAX_SAMPLES, len(test_ds))
    test_samples = [test_ds[i] for i in range(num_samples)]

    # 샘플을 GPU 개수만큼 분할
    num_gpus = len(gpu_ids)
    samples_per_gpu = len(test_samples) // num_gpus
    remainder = len(test_samples) % num_gpus

    sample_chunks = []
    start_idx = 0
    for i in range(num_gpus):
        # 나머지를 앞쪽 GPU들에 분배
        chunk_size = samples_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        sample_chunks.append(test_samples[start_idx:end_idx])
        start_idx = end_idx

    print(f"[INFO] 총 {len(test_samples)}개 샘플을 {num_gpus}개 GPU로 분할:")
    for i, chunk in enumerate(sample_chunks):
        print(f"  GPU {gpu_ids[i]}: {len(chunk)}개 샘플")

    # 멀티프로세싱으로 병렬 실행
    manager = Manager()
    results_dict = manager.dict()

    # 입출력 저장용 리스트 (save_io 플래그가 있을 때만)
    io_list = manager.list() if args.save_io else None

    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, model_name, few_shot_examples, sample_chunks[i], results_dict, i, args.save_io, io_list)
        )
        p.start()
        processes.append(p)

    # 모든 프로세스 완료 대기
    for p in processes:
        p.join()

    # 결과 집계
    total_correct = 0
    total_samples = 0
    for worker_id in range(num_gpus):
        result = results_dict[worker_id]
        total_correct += result["correct"]
        total_samples += result["total"]
        print(f"[결과] Worker {worker_id}: {result['correct']}/{result['total']}")

    acc = total_correct / total_samples if total_samples > 0 else 0.0

    # 결과 저장
    log_lines = []
    log_lines.append(f"Model : {model_name}\n")
    log_lines.append(f"GPUs : {gpu_ids}\n")
    log_lines.append(f"Few-shot per label : {K_PER_LABEL}\n")
    log_lines.append("\n==== Final Result ====\n")
    log_lines.append(f"Accuracy : {acc:.4f}\n")
    log_lines.append(f"Correct : {total_correct}/{total_samples}\n")

    results_dir = Path("/home/eastj/study/capstone/result")
    results_dir.mkdir(exist_ok = True)

    result_filename = args.result_name if args.result_name else f"{model_name}_result.txt"
    out_path = results_dir / result_filename
    with out_path.open("w", encoding = "utf-8") as f:
        f.writelines(log_lines)

    print(f'\n[INFO] 최종 정확도: {acc:.4f} ({total_correct}/{total_samples})')
    print(f'[INFO] 결과가 저장되었습니다: {out_path}')

    # 입출력 JSON 저장
    if args.save_io and io_list is not None:
        io_filename = args.io_output_file if args.io_output_file else f"{model_name}_io.json"
        io_path = results_dir / io_filename

        # Manager.list()를 일반 list로 변환
        io_data = list(io_list)

        with io_path.open("w", encoding="utf-8") as f:
            json.dump(io_data, f, ensure_ascii=False, indent=2)

        print(f'[INFO] 입출력 데이터가 저장되었습니다: {io_path}')
        print(f'[INFO] 총 {len(io_data)}개 샘플의 입출력이 기록되었습니다.')


if __name__ == "__main__":
    main()