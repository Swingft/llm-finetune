#!/bin/bash
# LoRA 어댑터 학습 워크플로우: 1. 의존성 설치 -> 2. LoRA 어댑터 학습

# 스크립트 실행 중 오류 발생 시 즉시 중단
set -e

echo "🚀 [LoRA 어댑터] 학습 워크플로우를 시작합니다."
echo "=================================================="

# --- 1단계: Python 의존성 설치 ---
echo "✅ 1. Python 의존성을 설치합니다..."
pip install torch transformers datasets peft trl bitsandbytes accelerate einops tiktoken
echo "--------------------------------------------------"


# --- 2단계: LoRA 어댑터 파인튜닝 ---
echo "✅ 2. LoRA 어댑터 파인튜닝을 시작합니다 (train_lora_adapter.py)..."
python train_lora_adapter.py
echo "--------------------------------------------------"

# 학습 완료 후 생성된 최종 어댑터 폴더 경로를 찾습니다.
LATEST_ADAPTER_DIR=$(find ./out -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)


echo "🎉 LoRA 어댑터 학습이 성공적으로 완료되었습니다!"
echo "=================================================="
echo "📁 최종 어댑터 경로: ${LATEST_ADAPTER_DIR}"
echo ""
echo "💡 다음 단계:"
echo "   아래 명령어를 실행하여 방금 학습한 어댑터를 GGUF로 변환할 수 있습니다."
echo "   chmod +x convert_lora_to_gguf.sh"
echo "   ./convert_lora_to_gguf.sh \"${LATEST_ADAPTER_DIR}\""
echo "=================================================="

# ==================================================
# ## ⚙️ Jupyter Notebook 실행 가이드
# ==================================================
#
# 1. 사전 준비:
#    아래 4개 파일이 모두 이 스크립트와 동일한 위치에 있어야 합니다.
#    - run_train_adapter_workflow.sh     (👈 이 파일)
#    - train_lora_adapter.py             (어댑터 학습 스크립트)
#    - convert_lora_to_gguf.sh           (어댑터 GGUF 변환 스크립트)
#    - train.jsonl                       (학습 데이터셋)
#
# 2. 실행 방법 (Jupyter Notebook 셀):
#    # (최초 한 번만) 실행 권한 부여
#    !chmod +x run_train_adapter_workflow.sh
#
#    # 어댑터 학습 워크플로우 실행
#    !./run_train_adapter_workflow.sh
#
