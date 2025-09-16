#!/bin/bash
# 완전한 환경 리셋 스크립트 - 모든 호환성 문제 해결

echo "🚨 완전한 환경 리셋을 시작합니다..."
echo "이 과정은 5-10분 정도 소요됩니다."

# 1. 모든 관련 패키지 제거
echo "1️⃣ 기존 패키지 완전 제거..."
pip uninstall -y torch torchvision torchaudio transformers trl peft datasets pyarrow pandas numpy

# 2. pip 캐시 완전 삭제
echo "2️⃣ 캐시 완전 삭제..."
pip cache purge
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface

# 3. 핵심 패키지부터 순차적으로 안정 버전 설치
echo "3️⃣ 핵심 패키지 설치 (numpy, pandas)..."
pip install numpy==1.24.3 --no-cache-dir
pip install pandas==2.0.3 --no-cache-dir

echo "4️⃣ pyarrow 안정 버전 설치..."
pip install pyarrow==14.0.1 --no-cache-dir

echo "5️⃣ PyTorch 생태계 설치..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

echo "6️⃣ HuggingFace 생태계 설치..."
pip install transformers==4.36.0 --no-cache-dir
pip install datasets==2.14.6 --no-cache-dir
pip install accelerate==0.24.1 --no-cache-dir

echo "7️⃣ PEFT와 TRL 설치..."
pip install peft==0.7.1 --no-cache-dir
pip install trl==0.7.4 --no-cache-dir

echo "8️⃣ 추가 의존성 설치..."
pip install bitsandbytes sentencepiece psutil huggingface-hub --no-cache-dir

# 9. 설치 검증
echo "9️⃣ 설치 검증 중..."
python3 << 'EOF'
import sys
print("🔍 패키지 버전 확인:")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

try:
    import torchvision
    print(f"✅ torchvision: {torchvision.__version__}")
except ImportError as e:
    print(f"❌ torchvision: {e}")

try:
    import transformers
    print(f"✅ transformers: {transformers.__version__}")
except ImportError as e:
    print(f"❌ transformers: {e}")

try:
    import datasets
    print(f"✅ datasets: {datasets.__version__}")
except ImportError as e:
    print(f"❌ datasets: {e}")

try:
    import pyarrow as pa
    print(f"✅ pyarrow: {pa.__version__}")
    if hasattr(pa, 'PyExtensionType'):
        print("  ✅ PyExtensionType 지원")
    else:
        print("  ⚠️ PyExtensionType 미지원")
except ImportError as e:
    print(f"❌ pyarrow: {e}")

try:
    from trl import SFTTrainer, SFTConfig
    print("✅ TRL: import 성공")
except ImportError as e:
    print(f"❌ TRL: {e}")

try:
    from peft import LoraConfig, TaskType
    print("✅ PEFT: import 성공")
except ImportError as e:
    print(f"❌ PEFT: {e}")

print("\n🧪 빠른 호환성 테스트:")
try:
    # torchvision 호환성 테스트
    import torch
    from transformers import AutoTokenizer
    
    # 간단한 토크나이저 로드 테스트
    print("  🔧 토크나이저 로드 테스트...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", trust_remote_code=True)
    print("  ✅ 토크나이저 로드 성공")
    
    # datasets 로드 테스트
    print("  🔧 datasets 로드 테스트...")
    from datasets import Dataset
    test_data = Dataset.from_dict({"text": ["test"]})
    print("  ✅ datasets 로드 성공")
    
    print("\n🎉 모든 호환성 테스트 통과!")
    
except Exception as e:
    print(f"  ❌ 호환성 테스트 실패: {e}")
    print("  💡 일부 패키지에 여전히 문제가 있을 수 있습니다.")

EOF

echo ""
echo "🎉 환경 리셋 완료!"
echo "💡 이제 학습 스크립트를 다시 실행해보세요."
echo ""
echo "📋 설치된 주요 버전:"
echo "  - PyTorch: 2.0.1"
echo "  - transformers: 4.36.0" 
echo "  - datasets: 2.14.6"
echo "  - pyarrow: 14.0.1"
echo "  - trl: 0.7.4"
echo "  - peft: 0.7.1"