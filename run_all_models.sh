#!/bin/bash

# 모든 모델을 순차적으로 실행하는 스크립트
# 사용법: bash run_all_models.sh

echo "======================================"
echo "모든 모델 실험 시작"
echo "======================================"
echo ""

# 실험 설정
SHOTS=0
SAMPLES=2000
USE_KOREAN="--use_korean"
GPUS="0,1,2"
DEBUG=""  # 디버그 모드: "--debug" 입력 시 샘플 1개만 테스트 (SAMPLES가 인덱스로 사용됨)
SAVE_IO="--save_io"  # 입출력 JSON 저장: "--save_io" 입력 시 활성화

# 모델 리스트
MODELS=("qwen" "llama" "ministral" "solar" "gpt")

# 각 모델 실행
for MODEL in "${MODELS[@]}"; do
    echo "--------------------------------------"
    echo "모델: $MODEL 실행 중..."
    echo "--------------------------------------"

    RESULT_NAME="${MODEL}_korean_${SHOTS}shot_${SAMPLES}.txt"
    IO_FILE="${MODEL}_io.json"

    python3 train.py \
        --model_name $MODEL \
        $USE_KOREAN \
        --shots $SHOTS \
        --samples $SAMPLES \
        --gpus $GPUS \
        --result_name $RESULT_NAME \
        $DEBUG \
        $SAVE_IO \
        $([ -n "$SAVE_IO" ] && echo "--io_output_file $IO_FILE" || echo "")

    if [ $? -eq 0 ]; then
        echo "✅ $MODEL 완료!"
    else
        echo "❌ $MODEL 실패!"
    fi

    echo ""
done

echo "======================================"
echo "모든 실험 완료!"
echo "======================================"
echo ""
echo "결과 파일:"
ls -lh result/*_korean_${SHOTS}shot_${SAMPLES}.txt 2>/dev/null || echo "결과 파일 없음"
echo ""
echo "결과 요약:"
for MODEL in "${MODELS[@]}"; do
    RESULT_FILE="result/${MODEL}_korean_${SHOTS}shot_${SAMPLES}.txt"
    if [ -f "$RESULT_FILE" ]; then
        echo -n "$MODEL: "
        grep "Accuracy" "$RESULT_FILE" | tail -1
    fi
done
