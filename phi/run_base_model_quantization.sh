#!/bin/bash
# 베이스 모델 양자화 전용 워크플로우: 의존성 설치 -> 베이스 모델 다운로드 및 GGUF 변환

set -e

echo "🚀 베이스 모델 양자화 워크플로우를 시작합니다."
echo "=================================================="

# --- 1단계: 필수 의존성 설치 ---
echo "✅ 1. 필수 의존성을 설치합니다..."

# Python 패키지 설치
pip install torch transformers datasets huggingface-hub

# 시스템 패키지 설치 (이미 설치되어 있을 가능성이 높지만 확인)
if command -v apt-get &> /dev/null; then
    echo "📦 시스템 패키지 확인 중..."
    apt-get update -qq
    apt-get install -y cmake build-essential libcurl4-openssl-dev
elif command -v yum &> /dev/null; then
    echo "📦 yum 패키지 매니저 감지. cmake와 libcurl-devel 설치 중..."
    yum install -y cmake libcurl-devel
elif command -v dnf &> /dev/null; then
    echo "📦 dnf 패키지 매니저 감지. cmake와 libcurl-devel 설치 중..."
    dnf install -y cmake libcurl-devel
else
    echo "⚠️ 지원되는 패키지 매니저를 찾을 수 없습니다."
    echo "cmake와 libcurl이 이미 설치되어 있다고 가정합니다."
fi

echo "✅ 의존성 설치 완료"
echo "--------------------------------------------------"

# --- 2단계: 베이스 모델 선택 ---
echo "✅ 2. 베이스 모델을 선택합니다..."

# 사용자가 매개변수로 모델을 지정했는지 확인
if [ -z "$1" ]; then
    echo "사용 가능한 베이스 모델들:"
    echo "  1. microsoft/Phi-3-mini-128k-instruct (기본값)"
    echo "  2. microsoft/Phi-3.5-mini-instruct"
    echo "  3. microsoft/Phi-3-medium-128k-instruct"
    echo "  4. meta-llama/Meta-Llama-3.1-8B-Instruct"
    echo ""
    echo "기본 모델(Phi-3-mini-128k-instruct)을 사용합니다."
    MODEL_ID="microsoft/Phi-3-mini-128k-instruct"
else
    MODEL_ID="$1"
    echo "지정된 모델: $MODEL_ID"
fi

echo "📁 선택된 모델: $MODEL_ID"
echo "--------------------------------------------------"

# --- 3단계: 베이스 모델 변환 스크립트 확인 ---
echo "✅ 3. 베이스 모델 변환 스크립트를 준비합니다..."

if [ ! -f "convert_base_model_to_gguf.sh" ]; then
    echo "❌ 베이스 모델 변환 스크립트 'convert_base_model_to_gguf.sh'를 찾을 수 없습니다."
    echo "현재 디렉토리의 스크립트 파일들:"
    ls -la *.sh 2>/dev/null || echo "쉘 스크립트 파일이 없습니다."
    exit 1
fi

chmod +x convert_base_model_to_gguf.sh
echo "✅ 베이스 모델 변환 스크립트 준비 완료"
echo "--------------------------------------------------"

# --- 4단계: 베이스 모델 GGUF 변환 및 양자화 ---
echo "✅ 4. 베이스 모델 GGUF 변환 및 양자화를 시작합니다..."
echo "   모델: $MODEL_ID"
echo "   예상 소요 시간: 10-30분 (모델 크기 및 시스템 성능에 따라)"
echo ""

./convert_base_model_to_gguf.sh "$MODEL_ID"

echo "--------------------------------------------------"

# --- 5단계: 결과 확인 및 정리 ---
echo "✅ 5. 생성된 GGUF 파일들을 확인합니다..."

if [ -d "/workspace/gguf_models" ]; then
    echo ""
    echo "📁 생성된 베이스 모델 GGUF 파일들:"
    ls -lh /workspace/gguf_models/*-base-*.gguf | awk '{print "   📄 " $9 " (" $5 ")"}'

    # 총 크기 계산
    TOTAL_SIZE=$(du -sh /workspace/gguf_models/ | cut -f1)
    echo ""
    echo "📊 gguf_models 디렉토리 총 크기: $TOTAL_SIZE"

    # 추천 모델 표시
    echo ""
    echo "💡 베이스 모델 추천 사용법:"
    if ls /workspace/gguf_models/*-base-q4_k_m.gguf 1> /dev/null 2>&1; then
        Q4_FILE=$(ls /workspace/gguf_models/*-base-q4_k_m.gguf | head -1)
        Q4_SIZE=$(du -h "$Q4_FILE" | cut -f1)
        echo "   🎯 균형잡힌 성능: $(basename "$Q4_FILE") ($Q4_SIZE)"
    fi

    if ls /workspace/gguf_models/*-base-q4_0.gguf 1> /dev/null 2>&1; then
        Q4_0_FILE=$(ls /workspace/gguf_models/*-base-q4_0.gguf | head -1)
        Q4_0_SIZE=$(du -h "$Q4_0_FILE" | cut -f1)
        echo "   ⚡ 빠른 추론: $(basename "$Q4_0_FILE") ($Q4_0_SIZE)"
    fi

    if ls /workspace/gguf_models/*-base-q5_k_m.gguf 1> /dev/null 2>&1; then
        Q5_FILE=$(ls /workspace/gguf_models/*-base-q5_k_m.gguf | head -1)
        Q5_SIZE=$(du -h "$Q5_FILE" | cut -f1)
        echo "   🏆 고품질: $(basename "$Q5_FILE") ($Q5_SIZE)"
    fi

else
    echo "❌ GGUF 모델 디렉토리를 찾을 수 없습니다."
    echo "변환 과정에서 문제가 발생했을 수 있습니다."
    exit 1
fi

echo ""
echo "🎉 베이스 모델 양자화가 성공적으로 완료되었습니다!"
echo "=================================================="
echo "📝 완료된 작업:"
echo "   ✅ $MODEL_ID 다운로드"
echo "   ✅ 토크나이저 문제 해결"
echo "   ✅ FP16 GGUF 변환"
echo "   ✅ 5가지 양자화 레벨 생성 (Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0)"
echo ""
echo "🔍 파인튜닝 모델과 비교:"
echo "   베이스 모델: *-base-*.gguf"
echo "   파인튜닝 모델: merged_model-finetuned-*.gguf"
echo ""
echo "🖥️ 사용 방법:"
echo "   1. 생성된 .gguf 파일을 로컬 머신으로 다운로드"
echo "   2. llama.cpp 또는 Ollama 등의 도구로 실행"
echo "   3. 예시: ./llama-cli -m model.gguf -p \"Your prompt here\""
echo ""
echo "📊 성능 비교 팁:"
echo "   동일한 프롬프트로 베이스 모델과 파인튜닝 모델을 테스트하여"
echo "   파인튜닝 효과를 확인해보세요."
echo "=================================================="

# ==================================================
# ## ⚙️ Jupyter Notebook 실행 가이드
# ==================================================
#
# 1. 사전 준비:
#    - run_base_model_quantization.sh (👈 이 파일)
#    - convert_base_model_to_gguf.sh (베이스 모델 변환 스크립트)
#
# 2. 실행 방법 (Jupyter Notebook 셀):
#    # 실행 권한 부여
#    !chmod +x run_base_model_quantization.sh
#    !chmod +x convert_base_model_to_gguf.sh
#
#    # 기본 모델 양자화
#    !./run_base_model_quantization.sh
#
#    # 다른 모델 양자화 예시
#    !./run_base_model_quantization.sh "microsoft/Phi-3.5-mini-instruct"