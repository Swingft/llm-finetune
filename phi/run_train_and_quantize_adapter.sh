#!/bin/bash
# LoRA 어댑터 학습 워크플로우: 의존성 설치 -> LoRA 어댑터 학습

set -e

echo "🚀 LoRA 어댑터 학습 워크플로우를 시작합니다."
echo "=================================================="

# --- 1단계: Python 의존성 설치 ---
echo "✅ 1. Python 의존성을 설치합니다..."
pip install torch transformers datasets peft trl bitsandbytes accelerate einops tiktoken huggingface-hub
echo "--------------------------------------------------"

# --- 2단계: 학습 데이터 파일 확인 ---
echo "✅ 2. 학습 데이터 파일을 확인합니다..."
if [ ! -f "json_data.jsonl" ]; then
    echo "❌ 학습 데이터 파일 'json_data.jsonl'을 찾을 수 없습니다."
    echo "현재 디렉토리의 파일들:"
    ls -la *.jsonl 2>/dev/null || echo "JSONL 파일이 없습니다."
    exit 1
fi
echo "✅ 학습 데이터 파일 확인 완료: json_data.jsonl"
echo "--------------------------------------------------"

# --- 3단계: LoRA 어댑터 학습 스크립트 확인 ---
echo "✅ 3. LoRA 어댑터 학습 스크립트를 확인합니다..."
if [ ! -f "train_lora_adapter.py" ]; then
    echo "❌ LoRA 학습 스크립트 'train_lora_adapter.py'를 찾을 수 없습니다."
    echo "현재 디렉토리의 파이썬 파일들:"
    ls -la *.py 2>/dev/null || echo "Python 파일이 없습니다."
    exit 1
fi
echo "✅ LoRA 어댑터 학습 스크립트 확인 완료"
echo "--------------------------------------------------"

# --- 4단계: LoRA 어댑터 파인튜닝 ---
echo "✅ 4. LoRA 어댑터 파인튜닝을 시작합니다..."
echo "   예상 소요 시간: 20-40분 (데이터 크기 및 하드웨어에 따라)"
echo ""
python train_lora_adapter.py
echo "--------------------------------------------------"

# --- 5단계: 생성된 어댑터 확인 ---
echo "✅ 5. 학습된 어댑터를 확인합니다..."

# 학습 완료 후 생성된 최종 어댑터 폴더 경로를 찾습니다.
LATEST_ADAPTER_DIR=$(find ./out -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)

if [ -z "$LATEST_ADAPTER_DIR" ]; then
    echo "❌ 학습된 어댑터 디렉토리를 찾을 수 없습니다."
    echo "out 디렉토리 내용:"
    ls -la ./out/ 2>/dev/null || echo "out 디렉토리가 없습니다."
    exit 1
fi

# 어댑터 파일 검증
if [ ! -f "$LATEST_ADAPTER_DIR/adapter_config.json" ]; then
    echo "❌ 유효하지 않은 어댑터 디렉토리: adapter_config.json이 없습니다."
    exit 1
fi

if [ ! -f "$LATEST_ADAPTER_DIR/adapter_model.safetensors" ] && [ ! -f "$LATEST_ADAPTER_DIR/adapter_model.bin" ]; then
    echo "❌ 어댑터 모델 파일이 없습니다."
    exit 1
fi

echo "✅ 어댑터 검증 완료: $LATEST_ADAPTER_DIR"
echo ""
echo "어댑터 파일 확인:"
ls -la "${LATEST_ADAPTER_DIR}/"

# 어댑터 크기 확인
ADAPTER_SIZE=$(du -sh "$LATEST_ADAPTER_DIR" | cut -f1)
echo ""
echo "📊 어댑터 크기: $ADAPTER_SIZE"

echo ""
echo "🎉 LoRA 어댑터 학습이 성공적으로 완료되었습니다!"
echo "=================================================="
echo "📁 최종 어댑터 경로: ${LATEST_ADAPTER_DIR}"
echo ""
echo "💡 다음 단계:"
echo "   1. 어댑터를 GGUF로 변환하려면:"
echo "      ./convert_lora_to_gguf.sh \"${LATEST_ADAPTER_DIR}\""
echo ""
echo "   2. 또는 두 단계를 한번에 실행하려면:"
echo "      chmod +x convert_lora_to_gguf.sh"
echo "      ./convert_lora_to_gguf.sh \"${LATEST_ADAPTER_DIR}\""
echo ""
echo "📝 참고사항:"
echo "   - 이 어댑터는 베이스 모델과 함께 사용해야 합니다"
echo "   - 베이스 모델: microsoft/Phi-3-mini-128k-instruct"
echo "   - 사용법: ./llama-cli -m base.gguf --lora adapter.gguf -p \"prompt\""
echo "=================================================="

# ==================================================
# ## ⚙️ Jupyter Notebook 실행 가이드
# ==================================================
#
# 1. 사전 준비:
#    - run_train_and_quantize_adapter.sh (👈 이 파일)
#    - train_lora_adapter.py (어댑터 학습 스크립트)
#    - convert_lora_to_gguf.sh (어댑터 GGUF 변환 스크립트)
#    - json_data.jsonl (학습 데이터셋)
#
# 2. 실행 방법 (Jupyter Notebook 셀):
#    # 실행 권한 부여
#    !chmod +x run_train_and_quantize_adapter.sh
#    !chmod +x convert_lora_to_gguf.sh
#
#    # 어댑터 학습 워크플로우 실행
#    !./run_train_and_quantize_adapter.sh