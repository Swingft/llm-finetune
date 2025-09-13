#!/bin/bash
# 개선된 GGUF 변환 스크립트 (모든 문제 해결된 최종 버전)

set -e # 에러 발생 시 스크립트 중단

echo "🚀 GGUF 변환 프로세스 시작"
echo "================================"

# --- 필수 패키지 자동 설치 (cmake, libcurl-dev) ---
echo "✅ 필수 패키지(cmake, libcurl-dev) 설치 확인 및 시도..."

# apt (Debian/Ubuntu) 또는 yum/dnf (CentOS/RHEL) 확인
if command -v apt-get &> /dev/null; then
    echo "📦 apt 패키지 매니저 감지. cmake와 libcurl4-openssl-dev 설치 중..."
    apt-get update
    apt-get install -y cmake libcurl4-openssl-dev
elif command -v yum &> /dev/null; then
    echo "📦 yum 패키지 매니저 감지. cmake와 libcurl-devel 설치 중..."
    yum install -y cmake libcurl-devel
elif command -v dnf &> /dev/null; then
    echo "📦 dnf 패키지 매니저 감지. cmake와 libcurl-devel 설치 중..."
    dnf install -y cmake libcurl-devel
else
    echo "⚠️ 지원되는 패키지 매니저(apt, yum, dnf)를 찾을 수 없습니다."
    echo "cmake와 libcurl-devel(또는 libcurl4-openssl-dev)가 설치되어 있는지 확인해주세요."
fi
echo "✅ 패키지 설치/확인 완료."
# --- 설치 로직 끝 ---


# 모델 경로 자동 감지 또는 수동 설정
if [ -z "$1" ]; then
    MODEL_PATH=$(find ./out -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
    if [ -z "$MODEL_PATH" ]; then
        echo "❌ ./out 디렉토리에서 학습 결과물을 찾을 수 없습니다."
        exit 1
    fi
    echo "✅ 가장 최근 모델을 자동으로 감지했습니다."
else
    MODEL_PATH="$1"
    echo "✅ 수동으로 모델 경로를 지정했습니다."
fi

echo "📁 모델 경로: $MODEL_PATH"

GGUF_DIR="./gguf_models"
mkdir -p "$GGUF_DIR"

if [ ! -d "llama.cpp" ]; then
    echo "📥 llama.cpp 클론 중..."
    git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp
echo "🔄 llama.cpp 업데이트 중..."
git pull

echo "🔨 llama.cpp 빌드 중 (CMake, CUDA 지원)..."
rm -rf build
if command -v nvcc &> /dev/null; then
    echo "🚀 CUDA 감지 - GPU 가속 빌드"
    cmake -B build -DLLAMA_CUDA=ON
else
    echo "⚠️ CUDA 미감지 - CPU 전용 빌드"
    cmake -B build
fi
cmake --build build --config Release -j $(nproc)

echo "📦 Python 의존성 설치 중..."
pip install -q -r requirements.txt

ABS_MODEL_PATH=$(cd ..; realpath "$MODEL_PATH")
if [ ! -f "$ABS_MODEL_PATH/config.json" ]; then
    echo "❌ 유효하지 않은 모델 디렉토리: $ABS_MODEL_PATH"
    exit 1
fi

MODEL_NAME=$(basename "$ABS_MODEL_PATH")
OUTPUT_NAME="${MODEL_NAME}-finetuned"

echo "🔄 HuggingFace → GGUF 변환 중..."
# --- 수정된 부분: convert.py -> convert-hf-to-gguf.py ---
python convert-hf-to-gguf.py "$ABS_MODEL_PATH" \
    --outtype f16 \
    --outfile "../$GGUF_DIR/${OUTPUT_NAME}-f16.gguf"
# --- 수정 끝 ---

cd ..
echo "✅ FP16 변환 완료"

cd "$GGUF_DIR"
FP16_SIZE=$(du -h "${OUTPUT_NAME}-f16.gguf" | cut -f1)
echo "📏 FP16 모델 크기: $FP16_SIZE"

echo "🔄 양자화 시작..."
declare -A QUANT_LEVELS=(
    ["q4_k_m"]="4비트, 균형잡힌 품질 (추천)"
    ["q5_k_m"]="5비트, 높은 품질"
)

for level in "${!QUANT_LEVELS[@]}"; do
    output_file="${OUTPUT_NAME}-${level}.gguf"
    echo "    🔧 $level 양자화 중... (${QUANT_LEVELS[$level]})"

    ../llama.cpp/build/bin/quantize \
        "${OUTPUT_NAME}-f16.gguf" \
        "$output_file" \
        "$level"

    size=$(du -h "$output_file" | cut -f1)
    echo "    ✅ $output_file 생성 완료 ($size)"
done

echo ""
echo "🎉 GGUF 변환 완료!"
echo "================================"
echo "생성된 모델 파일들:"
ls -lh *.gguf | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "💡 추천: ${OUTPUT_NAME}-q4_k_m.gguf"
echo ""
tar -czf "${MODEL_NAME}-gguf-models.tar.gz" *.gguf
echo "✅ 압축 완료: ${MODEL_NAME}-gguf-models.tar.gz"
echo "================================"
