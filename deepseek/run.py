#!/usr/bin/env python3
"""
맥북에서 GGUF 모델 실행 스크립트
Intel Mac과 M1/M2/M3 Mac 모두 지원
"""

import subprocess
import sys
import os
from pathlib import Path


def install_llama_cpp_python():
    """llama-cpp-python 설치 (Metal 가속 포함)"""
    print("📦 llama-cpp-python 설치 중...")

    # M칩 맥의 경우 Metal 가속 활성화
    env_vars = os.environ.copy()
    if "arm64" in subprocess.check_output(["uname", "-m"]).decode().lower():
        env_vars["CMAKE_ARGS"] = "-DLLAMA_METAL=on"
        env_vars["FORCE_CMAKE"] = "1"
        print("🚀 M칩 맥 감지 - Metal 가속 활성화")

    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "llama-cpp-python", "--upgrade", "--force-reinstall", "--no-cache-dir"
    ], env=env_vars)


def run_inference_example():
    """GGUF 모델로 추론 실행 예제"""
    from llama_cpp import Llama

    # GGUF 모델 파일 경로 (실제 경로로 수정 필요)
    model_path = "./phi-finetuned-q4_k_m.gguf"  # Q4_K_M 추천 (속도/품질 균형)

    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("GGUF 파일을 현재 디렉토리에 배치하고 경로를 수정해주세요.")
        return

    print(f"🔄 모델 로딩 중: {model_path}")

    # 모델 초기화
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,  # 컨텍스트 길이
        n_threads=None,  # 자동으로 최적 스레드 수 설정
        n_gpu_layers=-1,  # M칩: Metal 가속, Intel: CPU만 사용
        verbose=False
    )

    print("✅ 모델 로딩 완료!")

    # 추론 테스트
    while True:
        user_input = input("\n사용자 입력 (종료: quit): ")
        if user_input.lower() == 'quit':
            break

        # Phi 모델 포맷에 맞춰 프롬프트 구성
        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"

        print("🤔 생성 중...")
        response = llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stop=["### Instruction:", "### Response:"]
        )

        print(f"🤖 응답: {response['choices'][0]['text'].strip()}")


def benchmark_model():
    """모델 성능 벤치마크"""
    from llama_cpp import Llama
    import time

    model_path = "./phi-finetuned-q4_k_m.gguf"

    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return

    print("🚀 벤치마크 시작...")

    start_time = time.time()
    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
    load_time = time.time() - start_time

    # 추론 속도 테스트
    test_prompt = "### Instruction:\n안녕하세요, 자기소개를 해주세요.\n\n### Response:\n"

    start_time = time.time()
    response = llm(test_prompt, max_tokens=100, temperature=0.7)
    inference_time = time.time() - start_time

    tokens_generated = len(response['choices'][0]['text'].split())
    tokens_per_sec = tokens_generated / inference_time

    print(f"📊 벤치마크 결과:")
    print(f"   모델 로딩 시간: {load_time:.2f}초")
    print(f"   추론 시간: {inference_time:.2f}초")
    print(f"   생성 토큰 수: {tokens_generated}")
    print(f"   토큰/초: {tokens_per_sec:.2f}")


if __name__ == "__main__":
    print("🍎 맥OS GGUF 모델 실행기")
    print("=" * 40)

    try:
        import llama_cpp

        print("✅ llama-cpp-python 이미 설치됨")
    except ImportError:
        install_llama_cpp_python()

    while True:
        print("\n선택하세요:")
        print("1. 대화형 추론")
        print("2. 성능 벤치마크")
        print("3. 종료")

        choice = input("선택 (1-3): ").strip()

        if choice == "1":
            run_inference_example()
        elif choice == "2":
            benchmark_model()
        elif choice == "3":
            break
        else:
            print("잘못된 선택입니다.")