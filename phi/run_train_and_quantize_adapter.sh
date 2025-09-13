#!/bin/bash
# LoRA 어댑터 학습 및 GGUF 변환 워크플로우 (v5.0: 의존성 충돌 해결)

set -e

echo "🚀 LoRA 어댑터 학습 및 GGUF 변환 워크플로우를 시작합니다. (v5.0)"
echo "=================================================="

# --- 1단계: Python 의존성 설치 ---
echo "✅ 1. Python 의존성을 안정적인 방식으로 설치합니다..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft trl bitsandbytes accelerate einops tiktoken huggingface-hub
echo "✅ 의존성 설치 완료."
echo "--------------------------------------------------"

# --- 2단계: 필수 파일 확인 ---
echo "✅ 2. 필수 파일들을 확인합니다..."
if [ ! -f "json_data.jsonl" ]; then
    echo "❌ 학습 데이터 파일 'json_data.jsonl'을 찾을 수 없습니다."
    exit 1
fi
if [ ! -f "train_lora_adapter.py" ]; then
    echo "❌ LoRA 학습 스크립트 'train_lora_adapter.py'를 찾을 수 없습니다."
    exit 1
fi
if [ ! -f "convert_lora_to_gguf.sh" ]; then
    echo "❌ GGUF 변환 스크립트 'convert_lora_to_gguf.sh'를 찾을 수 없습니다."
    exit 1
fi
echo "✅ 모든 필수 파일 확인 완료."
echo "--------------------------------------------------"

# --- 3단계: 기존 학습 결과 확인 및 파인튜닝 ---
echo "✅ 3. 기존 학습 결과를 확인합니다..."
mkdir -p ./out
LATEST_ADAPTER_DIR=$(find ./out -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)

if [ -n "$LATEST_ADAPTER_DIR" ] && [ -f "$LATEST_ADAPTER_DIR/adapter_config.json" ]; then
    echo "✅ 이미 학습된 유효한 어댑터를 찾았습니다."
    echo "   경로: $LATEST_ADAPTER_DIR"
    echo "   >> 파인튜닝 단계를 건너뜁니다."
    echo "--------------------------------------------------"
else
    echo "ℹ️  기존 학습 결과를 찾을 수 없거나 유효하지 않습니다. 새로 파인튜닝을 시작합니다."
    echo "--------------------------------------------------"
    echo "🚀 3-1. LoRA 어댑터 파인튜닝을 시작합니다..."
    echo "   예상 소요 시간: 20-40분 (데이터 크기 및 하드웨어에 따라)"
    python train_lora_adapter.py
    echo "--------------------------------------------------"
    LATEST_ADAPTER_DIR=$(find ./out -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
    if [ ! -f "$LATEST_ADAPTER_DIR/adapter_config.json" ]; then
        echo "❌ 학습 후 유효한 어댑터를 찾지 못했습니다."
        exit 1
    fi
fi

echo "✅ 최종 어댑터 확인 완료: $LATEST_ADAPTER_DIR"
ADAPTER_SIZE=$(du -sh "$LATEST_ADAPTER_DIR" | cut -f1)
echo "📊 어댑터 크기: $ADAPTER_SIZE"
echo "--------------------------------------------------"

# --- 4단계: GGUF 변환 스크립트 수정 ---
echo "✅ 4. 의존성 충돌 방지를 위해 GGUF 변환 스크립트를 수정합니다..."
# convert_lora_to_gguf.sh 내부의 'pip install' 명령어를 주석 처리하여,
# 이미 설치된 안정적인 라이브러리 버전이 변경되지 않도록 합니다.
sed -i "s/pip install -q -r requirements.txt/# pip install -q -r requirements.txt/" "convert_lora_to_gguf.sh"
echo "✅ GGUF 변환 스크립트 수정 완료."
echo "--------------------------------------------------"


# --- 5단계: LoRA 어댑터 GGUF 양자화 ---
echo "✅ 5. LoRA 어댑터를 GGUF로 변환합니다..."
echo ""
# GGUF 변환 스크립트 실행
chmod +x convert_lora_to_gguf.sh
./convert_lora_to_gguf.sh "${LATEST_ADAPTER_DIR}"
echo "--------------------------------------------------"


echo ""
echo "🎉 전체 워크플로우가 성공적으로 완료되었습니다!"
echo "=================================================="
echo "📁 최종 LoRA 어댑터 경로: ${LATEST_ADAPTER_DIR}"
echo "   (GGUF 파일은 /workspace/gguf_models/ 디렉토리에서 확인하세요)"
echo ""
echo "📝 참고사항:"
echo "   - 생성된 GGUF 어댑터는 베이스 모델과 함께 사용해야 합니다."
echo "   - 베이스 모델: microsoft/Phi-3-mini-128k-instruct"
echo "   - 사용법 예시: ./llama-cli -m base.gguf --lora adapter.gguf -p \"prompt\""
echo "=================================================="

