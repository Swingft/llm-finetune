#!/bin/bash
set -e

MODEL_PATH="./out/Phi-3-mini-128k-instruct_merged_r128_sensitive/merged_model"
GGUF_DIR="/workspace/gguf_models"

echo "학습, 병합 및 GGUF 변환 프로세스 시작"
echo "실제 모델 경로: $MODEL_PATH"

# 수정된 체크 함수 - 분할된 safetensors 파일 지원
check_training_completed() {
    if [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ] && [ -f "$MODEL_PATH/tokenizer.json" ]; then
        # 분할된 safetensors 파일도 체크
        if [ -f "$MODEL_PATH/pytorch_model.bin" ] || [ -f "$MODEL_PATH/model.safetensors" ] || [ -f "$MODEL_PATH/model-00001-of-00002.safetensors" ]; then
            return 0  # 학습 완료
        fi
    fi
    return 1  # 학습 미완료
}

if check_training_completed; then
    echo "✅ 학습이 이미 완료되어 있어 건너뜁니다!"
else
    echo "🚀 LoRA 학습 및 병합 시작..."
    python train_and_merge_lora.py
fi

echo "🔄 GGUF 변환 및 양자화 시작..."
bash convert_merged_model_to_gguf.sh

echo "🎉 모든 프로세스 완료!"
