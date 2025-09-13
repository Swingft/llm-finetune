#!/bin/bash
# LoRA 어댑터 전용 GGUF 변환 스크립트

set -e # 에러 발생 시 스크립트 중단

echo "🚀 [LoRA 어댑터] GGUF 변환 프로세스 시작"
echo "================================"

# LoRA 어댑터 경로 확인
if [ -z "$1" ]; then
    echo "❌ 변환할 LoRA 어댑터 디렉토리 경로를 입력해주세요."
    echo "사용법: $0 [LoRA_어댑터_경로]"
    echo "예시: $0 ./out/Phi-3.5-mini-instruct_with_lora_r128_JSON/lora_adapter"
    exit 1
fi

LORA_PATH="$1"

echo "📁 LoRA 경로: $LORA_PATH"

# llama.cpp가 빌드되었는지 확인
if [ ! -f "llama.cpp/build/bin/quantize" ]; then
    echo "⚠️ llama.cpp가 빌드되지 않았습니다. 먼저 전체 모델 변환 스크립트를 실행하여 빌드해주세요."
    exit 1
fi

cd llama.cpp

# Python 패키지 설치 (이미 되어있을 가능성 높음)
echo "📦 Python 의존성 설치 확인 중..."
pip install -q -r requirements.txt

ABS_LORA_PATH=$(cd ..; realpath "$LORA_PATH")
if [ ! -f "$ABS_LORA_PATH/adapter_config.json" ]; then
    echo "❌ 유효하지 않은 LoRA 어댑터 디렉토리: $ABS_LORA_PATH"
    exit 1
fi

# LoRA 어댑터 GGUF 변환 스크립트 실행
# 결과물은 입력된 LoRA 폴더 내에 ggml-adapter-model.bin 파일로 저장됨
echo "🔄 LoRA 어댑터 → GGUF 변환 중..."
python convert-lora-to-gguf.py "$ABS_LORA_PATH"

# 결과 파일 이름 변경 및 이동
LORA_NAME=$(basename "$ABS_LORA_PATH")
OUTPUT_FILE="../gguf_models/${LORA_NAME}.bin"

mv "$ABS_LORA_PATH/ggml-adapter-model.bin" "$OUTPUT_FILE"

cd ..
echo "✅ LoRA GGUF 변환 완료"

size=$(du -h "$OUTPUT_FILE" | cut -f1)

echo ""
echo "🎉 LoRA GGUF 변환 완료!"
echo "================================"
echo "생성된 LoRA 파일: $OUTPUT_FILE ($size)"
echo ""
echo "💡 사용법:"
echo "   베이스 GGUF 모델과 함께 --lora 옵션으로 이 파일을 지정하여 사용하세요."
echo "   예: ./main -m base-model.gguf --lora ${LORA_NAME}.bin"
echo "================================"
