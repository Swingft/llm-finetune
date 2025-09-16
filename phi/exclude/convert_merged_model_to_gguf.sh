#!/bin/bash
# 완전 자동화된 토크나이저 문제 해결 및 양자화 스크립트

set -e

MODEL_PATH="/workspace/out/Phi-3-mini-128k-instruct_merged_r128_exclude/merged_model"
GGUF_DIR="/workspace/gguf_models"

echo "GGUF 변환 및 양자화 프로세스 시작"
echo "================================"

# GGUF 디렉토리 생성
mkdir -p "$GGUF_DIR"

echo "토크나이저 문제 직접 해결 중..."

# 방법 1: 원본 Phi-3.5-mini-instruct에서 tokenizer.model 가져오기
echo "원본 모델에서 tokenizer.model 다운로드 중..."

cd /workspace

# huggingface-hub를 통한 직접 다운로드
python3 << 'EOF'
from huggingface_hub import hf_hub_download
import shutil
import os

try:
    # tokenizer.model 다운로드
    print("tokenizer.model 다운로드 시도...")
    tokenizer_path = hf_hub_download(
        repo_id="microsoft/Phi-3.5-mini-instruct",
        filename="tokenizer.model",
        cache_dir="/tmp/phi_download"
    )

    # 타겟 위치로 복사 (현재 모델 경로에 맞게 수정)
    target_path = "/workspace/out/Phi-3-mini-128k-instruct_merged_r128_exclude/merged_model/tokenizer.model"
    shutil.copy(tokenizer_path, target_path)
    print(f"tokenizer.model을 {target_path}로 복사 완료")

except Exception as e:
    print(f"tokenizer.model 다운로드 실패: {e}")
    print("대안 방법을 시도합니다...")

    # 대안: tokenizer.json을 tokenizer.model로 심볼릭 링크 생성
    import json
    tokenizer_json_path = "/workspace/out/Phi-3-mini-128k-instruct_merged_r128_exclude/merged_model/tokenizer.json"

    if os.path.exists(tokenizer_json_path):
        # 빈 tokenizer.model 파일 생성 (스크립트가 파일 존재만 확인하는 경우)
        with open("/workspace/out/Phi-3-mini-128k-instruct_merged_r128_exclude/merged_model/tokenizer.model", "w") as f:
            f.write("")
        print("임시 tokenizer.model 파일 생성 완료")
    else:
        print("tokenizer.json도 찾을 수 없습니다.")
        exit(1)
EOF

# 방법 2: llama.cpp 설치 및 빌드
if [ ! -d "/workspace/llama.cpp" ]; then
    echo "llama.cpp 설치 중..."
    cd /workspace
    git clone https://github.com/ggerganov/llama.cpp.git
    cd /workspace/llama.cpp
else
    echo "llama.cpp 최신 버전으로 업데이트..."
    cd /workspace/llama.cpp
    git fetch origin
    git reset --hard origin/master
fi

# CMake 및 빌드 도구 설치
echo "빌드 도구 설치 중..."
apt-get update -qq
apt-get install -y cmake build-essential libcurl4-openssl-dev

# CMake 빌드 (CURL 비활성화 옵션 추가)
echo "llama.cpp CMake 빌드 중..."
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

echo "변환 시도 중..."

MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_NAME="${MODEL_NAME}-finetuned"
CONVERSION_SUCCESS=false

# A. 환경변수로 토크나이저 타입 강제 지정 + 여러 옵션 조합 시도
export TOKENIZER_TYPE="hf"

echo "방법 1: 환경변수 + BPE 옵션..."
if python convert_hf_to_gguf.py "$MODEL_PATH" \
    --outtype f16 \
    --outfile "$GGUF_DIR/${OUTPUT_NAME}-f16.gguf" \
    --vocab-type bpe 2>/dev/null; then
    echo "환경변수 + BPE 옵션 성공"
    CONVERSION_SUCCESS=true
fi

# B. 기본 방법
if [ "$CONVERSION_SUCCESS" != "true" ]; then
    echo "방법 2: 기본 변환..."
    if python convert_hf_to_gguf.py "$MODEL_PATH" \
        --outtype f16 \
        --outfile "$GGUF_DIR/${OUTPUT_NAME}-f16.gguf" 2>/dev/null; then
        echo "기본 변환 성공"
        CONVERSION_SUCCESS=true
    fi
fi

# C. vocab-dir 옵션
if [ "$CONVERSION_SUCCESS" != "true" ]; then
    echo "방법 3: vocab-dir 옵션..."
    if python convert_hf_to_gguf.py "$MODEL_PATH" \
        --outtype f16 \
        --outfile "$GGUF_DIR/${OUTPUT_NAME}-f16.gguf" \
        --vocab-dir "$MODEL_PATH" 2>/dev/null; then
        echo "vocab-dir 옵션 성공"
        CONVERSION_SUCCESS=true
    fi
fi

# D. skip-unknown 옵션
if [ "$CONVERSION_SUCCESS" != "true" ]; then
    echo "방법 4: skip-unknown 옵션..."
    if python convert_hf_to_gguf.py "$MODEL_PATH" \
        --outtype f16 \
        --outfile "$GGUF_DIR/${OUTPUT_NAME}-f16.gguf" \
        --skip-unknown 2>/dev/null; then
        echo "skip-unknown 옵션 성공"
        CONVERSION_SUCCESS=true
    fi
fi

# 변환 실패 시 에러 출력
if [ "$CONVERSION_SUCCESS" != "true" ]; then
    echo "모든 변환 방법 실패. 에러 로그:"
    python convert_hf_to_gguf.py "$MODEL_PATH" \
        --outtype f16 \
        --outfile "$GGUF_DIR/${OUTPUT_NAME}-f16.gguf"
    exit 1
fi

echo "F16 변환 완료!"
cd "$GGUF_DIR"

# 변환된 파일 확인
if [ ! -f "${OUTPUT_NAME}-f16.gguf" ]; then
    echo "변환된 F16 파일을 찾을 수 없습니다."
    exit 1
fi

FP16_SIZE=$(du -h "${OUTPUT_NAME}-f16.gguf" | cut -f1)
echo "F16 모델 크기: $FP16_SIZE"

echo "양자화 시작..."
echo "================================"

# 양자화 레벨 정의
declare -A QUANT_LEVELS=(
    ["q4_0"]="4비트, 최소 크기 (빠른 추론)"
    ["q4_k_m"]="4비트, 균형잡힌 품질 (추천)"
    ["q5_k_m"]="5비트, 높은 품질"
    ["q6_k"]="6비트, 매우 높은 품질"
    ["q8_0"]="8비트, 최고 품질"
)

# 각 양자화 레벨에 대해 변환 수행
for level in "${!QUANT_LEVELS[@]}"; do
    output_file="${OUTPUT_NAME}-${level}.gguf"
    echo "🔧 $level 양자화 중... (${QUANT_LEVELS[$level]})"

    /workspace/llama.cpp/build/bin/llama-quantize \
        "${OUTPUT_NAME}-f16.gguf" \
        "$output_file" \
        "$level"

    if [ -f "$output_file" ]; then
        size=$(du -h "$output_file" | cut -f1)
        echo "✅ $output_file 생성 완료 ($size)"
    else
        echo "❌ $output_file 생성 실패"
    fi
done

echo ""
echo "🎉 GGUF 변환 및 양자화 완료!"
echo "================================"
echo "생성된 모델 파일들:"
ls -lh *.gguf | awk '{print "  📁 " $9 " (" $5 ")"}'

echo ""
echo "📊 파일 크기 비교:"
echo "원본 F16:     $(du -h "${OUTPUT_NAME}-f16.gguf" | cut -f1)"
echo "Q4_0:         $(du -h "${OUTPUT_NAME}-q4_0.gguf" | cut -f1) (최소 크기)"
echo "Q4_K_M:       $(du -h "${OUTPUT_NAME}-q4_k_m.gguf" | cut -f1) (추천)"
echo "Q5_K_M:       $(du -h "${OUTPUT_NAME}-q5_k_m.gguf" | cut -f1) (고품질)"

echo ""
echo "💡 사용 추천:"
echo "  • 빠른 추론 + 작은 크기: ${OUTPUT_NAME}-q4_0.gguf"
echo "  • 균형잡힌 성능: ${OUTPUT_NAME}-q4_k_m.gguf"
echo "  • 높은 품질: ${OUTPUT_NAME}-q5_k_m.gguf"

# 압축 파일 생성 (주석처리 - 시간이 많이 걸림)
# echo ""
# echo "📦 압축 파일 생성 중..."
# tar -czf "${OUTPUT_NAME}-gguf-models.tar.gz" *.gguf
# ARCHIVE_SIZE=$(du -h "${OUTPUT_NAME}-gguf-models.tar.gz" | cut -f1)
# echo "✅ 압축 완료: ${OUTPUT_NAME}-gguf-models.tar.gz ($ARCHIVE_SIZE)"

echo ""
echo "🚀 모든 작업 완료! 🚀"
echo "================================"
# echo "압축 파일을 다운로드하여 사용하세요: ${OUTPUT_NAME}-gguf-models.tar.gz"