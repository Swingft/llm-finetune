#!/bin/bash
# LoRA 어댑터 전용 GGUF 변환 스크립트

set -e

echo "LoRA 어댑터 GGUF 변환 프로세스 시작"
echo "================================"

# LoRA 어댑터 경로 확인
if [ -z "$1" ]; then
    echo "❌ 변환할 LoRA 어댑터 디렉토리 경로를 입력해주세요."
    echo "사용법: $0 [LoRA_어댑터_경로]"
    echo "예시: $0 ./out/Phi-3-mini-128k-instruct_with_lora_r128_sensitive"
    exit 1
fi

LORA_PATH="$1"
echo "LoRA 어댑터 경로: $LORA_PATH"

# 절대 경로 변환
ABS_LORA_PATH=$(realpath "$LORA_PATH")
echo "절대 경로: $ABS_LORA_PATH"

# LoRA 어댑터 파일 확인
if [ ! -f "$ABS_LORA_PATH/adapter_config.json" ]; then
    echo "❌ 유효하지 않은 LoRA 어댑터 디렉토리: $ABS_LORA_PATH"
    echo "adapter_config.json 파일이 존재하지 않습니다."
    exit 1
fi

if [ ! -f "$ABS_LORA_PATH/adapter_model.safetensors" ] && [ ! -f "$ABS_LORA_PATH/adapter_model.bin" ]; then
    echo "❌ LoRA 어댑터 모델 파일을 찾을 수 없습니다."
    echo "adapter_model.safetensors 또는 adapter_model.bin 파일이 필요합니다."
    exit 1
fi

echo "✅ LoRA 어댑터 파일 검증 완료"

# GGUF 출력 디렉토리 생성
GGUF_DIR="/workspace/gguf_models"
mkdir -p "$GGUF_DIR"

# llama.cpp 확인 및 빌드 (성공한 패턴 사용)
if [ ! -d "/workspace/llama.cpp" ]; then
    echo "llama.cpp 설치 중..."
    cd /workspace
    git clone https://github.com/ggerganov/llama.cpp.git
else
    echo "llama.cpp 최신 버전으로 업데이트..."
    cd /workspace/llama.cpp
    git fetch origin
    git reset --hard origin/master
fi

# CMake 및 빌드 도구 설치 (성공한 패턴 사용)
echo "빌드 도구 설치 중..."
apt-get update -qq
apt-get install -y cmake build-essential libcurl4-openssl-dev

# CMake 빌드 (성공한 패턴 사용)
echo "llama.cpp CMake 빌드 중..."
cd /workspace/llama.cpp
mkdir -p build
cd build
cmake .. -DLLAMA_CURL=OFF
make -j$(nproc)

# 빌드 확인
if [ ! -f "/workspace/llama.cpp/build/bin/llama-quantize" ]; then
    echo "❌ llama-quantize 빌드 실패"
    exit 1
fi
echo "✅ llama.cpp 빌드 완료"

# Python 패키지 재설치
echo "Python 의존성 재설치..."
cd /workspace/llama.cpp
pip install -q -r requirements.txt

# LoRA 변환 스크립트 확인
CONVERT_SCRIPT=""
if [ -f "convert_lora_to_gguf.py" ]; then
    CONVERT_SCRIPT="convert_lora_to_gguf.py"
elif [ -f "convert-lora-to-ggml.py" ]; then
    CONVERT_SCRIPT="convert-lora-to-ggml.py"
elif [ -f "convert_lora_to_ggml.py" ]; then
    CONVERT_SCRIPT="convert_lora_to_ggml.py"
else
    echo "❌ LoRA 변환 스크립트를 찾을 수 없습니다."
    echo "사용 가능한 스크립트를 확인 중..."
    ls -la convert*lora* 2>/dev/null || echo "LoRA 변환 스크립트가 없습니다."
    exit 1
fi

echo "✅ LoRA 변환 스크립트 발견: $CONVERT_SCRIPT"

LORA_NAME=$(basename "$ABS_LORA_PATH")
OUTPUT_FILE="$GGUF_DIR/${LORA_NAME}-lora.gguf"

echo "LoRA 어댑터 → GGUF 변환 중..."
echo "   입력: $ABS_LORA_PATH"
echo "   스크립트: $CONVERT_SCRIPT"
echo "   출력: $OUTPUT_FILE"

CONVERSION_SUCCESS=false

# 방법 1: --outfile 옵션과 함께 시도
echo "방법 1: --outfile 옵션 시도..."
if python "$CONVERT_SCRIPT" "$ABS_LORA_PATH" --outfile "$OUTPUT_FILE" 2>/dev/null; then
    echo "--outfile 옵션 성공"
    CONVERSION_SUCCESS=true
fi

# 방법 2: 기본 출력 경로 시도
if [ "$CONVERSION_SUCCESS" != "true" ]; then
    echo "방법 2: 기본 변환 시도..."
    if python "$CONVERT_SCRIPT" "$ABS_LORA_PATH" 2>/dev/null; then
        # 생성된 파일 찾기
        if [ -f "ggml-adapter-model.gguf" ]; then
            mv "ggml-adapter-model.gguf" "$OUTPUT_FILE"
            echo "기본 변환 성공"
            CONVERSION_SUCCESS=true
        elif [ -f "$ABS_LORA_PATH/ggml-adapter-model.bin" ]; then
            cp "$ABS_LORA_PATH/ggml-adapter-model.bin" "$OUTPUT_FILE"
            echo "기본 변환 성공 (bin 파일)"
            CONVERSION_SUCCESS=true
        fi
    fi
fi

# 방법 3: 출력 디렉토리 지정
if [ "$CONVERSION_SUCCESS" != "true" ]; then
    echo "방법 3: 출력 디렉토리 지정 시도..."
    if python "$CONVERT_SCRIPT" "$ABS_LORA_PATH" --outdir "$GGUF_DIR" 2>/dev/null; then
        # 생성된 파일 찾기
        GENERATED_FILE=$(find "$GGUF_DIR" -name "*lora*" -o -name "*adapter*" | head -1)
        if [ -n "$GENERATED_FILE" ]; then
            mv "$GENERATED_FILE" "$OUTPUT_FILE"
            echo "출력 디렉토리 지정 성공"
            CONVERSION_SUCCESS=true
        fi
    fi
fi

# 변환 실패 시 에러 로그 출력
if [ "$CONVERSION_SUCCESS" != "true" ]; then
    echo "모든 변환 방법 실패. 에러 로그:"
    python "$CONVERT_SCRIPT" "$ABS_LORA_PATH" --outfile "$OUTPUT_FILE"
    exit 1
fi

cd /workspace

# 변환 결과 확인
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "❌ 변환된 LoRA 파일을 찾을 수 없습니다: $OUTPUT_FILE"
    exit 1
fi

size=$(du -h "$OUTPUT_FILE" | cut -f1)

echo ""
echo "🎉 LoRA GGUF 변환 완료!"
echo "================================"
echo "생성된 LoRA 파일: $(basename "$OUTPUT_FILE") ($size)"
echo "파일 경로: $OUTPUT_FILE"
echo ""
echo "💡 사용 방법:"
echo "   베이스 GGUF 모델과 함께 --lora 옵션으로 이 파일을 사용하세요."
echo ""
echo "   예시 (llama.cpp 사용):"
echo "   ./llama-cli -m base-model.gguf --lora $(basename "$OUTPUT_FILE") -p \"Your prompt here\""
echo ""
echo "   예시 (Ollama 사용):"
echo "   ollama run base-model --adapter $(basename "$OUTPUT_FILE")"
echo ""
echo "📝 중요 참고사항:"
echo "   • LoRA 어댑터는 원본 베이스 모델과 함께 사용해야 합니다"
echo "   • 이 파일만으로는 독립적으로 실행할 수 없습니다"
echo "   • 베이스 모델: microsoft/Phi-3-mini-128k-instruct"
echo ""
echo "🔍 파일 비교:"
echo "   베이스 모델: ~2-7GB (양자화 수준에 따라)"
echo "   LoRA 어댑터: $size (매우 작음)"
echo "   병합된 모델: ~2-7GB (베이스 모델과 유사)"
echo "================================"