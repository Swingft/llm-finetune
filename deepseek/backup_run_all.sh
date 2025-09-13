#!/bin/bash
# 전체 워크플로우 실행 스크립트: 1. 의존성 설치 -> 2. 학습 -> 3. GGUF 변환

# 스크립트 실행 중 오류 발생 시 즉시 중단
set -e

echo "🚀 전체 워크플로우를 시작합니다."
echo "=================================================="

# --- 1단계: Python 의존성 설치 ---
echo "✅ 1. Python 의존성을 설치합니다..."
pip install torch transformers datasets peft trl bitsandbytes accelerate einops tiktoken
echo "--------------------------------------------------"


# --- 2단계: 모델 파인튜닝 ---
echo "✅ 2. 모델 파인튜닝을 시작합니다 (train_phi.py)..."
python train_lora_adapter.py
echo "--------------------------------------------------"


# --- 3단계: GGUF 변환 ---
echo "✅ 3. GGUF 변환을 시작합니다..."

# train_lora_adapter.py 실행 후 out/ 디렉토리에 생성된 가장 최근 폴더를 찾습니다.
LATEST_TRAIN_DIR=$(find ./out -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)

if [ -z "$LATEST_TRAIN_DIR" ]; then
    echo "❌ ./out 디렉토리에서 학습 결과물 폴더를 찾지 못했습니다."
    exit 1
fi

# 학습 결과물 폴더 내의 merged_model 경로를 최종 경로로 지정합니다.
MERGED_MODEL_PATH="${LATEST_TRAIN_DIR}/merged_model"

if [ ! -d "$MERGED_MODEL_PATH" ]; then
    echo "❌ 병합된 모델 폴더를 찾지 못했습니다: ${MERGED_MODEL_PATH}"
    echo "➡️ train_phi.py 스크립트가 성공적으로 모델을 병합했는지 확인해주세요."
    exit 1
fi

echo "📁 GGUF로 변환할 모델 경로: ${MERGED_MODEL_PATH}"

# convert_merged_model_to_gguf.sh 스크립트에 실행 권한 부여
chmod +x convert_to_gguf.sh

# GGUF 변환 스크립트를 올바른 경로와 함께 실행
./convert_to_gguf.sh "$MERGED_MODEL_PATH"
echo "--------------------------------------------------"

echo "🎉 모든 워크플로우가 성공적으로 완료되었습니다!"
echo "=================================================="


# ==================================================
# ## ⚙️ Jupyter Notebook 실행 가이드
# ==================================================
#
# 1. 사전 준비:
#    아래 4개 파일이 모두 이 스크립트와 동일한 위치에 있어야 합니다.
#    - backup_run_all.sh            (👈 이 파일)
#    - train_lora_adapter.py          (모델 학습 스크립트)
#    - convert_merged_model_to_gguf.sh    (GGUF 변환 스크립트)
#    - train.jsonl           (학습 데이터셋)
#
# 2. 실행 방법 (Jupyter Notebook 셀):
#    아래 두 줄의 명령어를 순서대로 실행하세요.
#
#    # (최초 한 번만) 스크립트에 실행 권한 부여
#    !chmod +x backup_run_all.sh
#
#    # 전체 워크플로우 실행
#    !./backup_run_all.sh