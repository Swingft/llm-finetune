#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from transformers import AutoTokenizer


def load_gguf_model():
    """GGUF 모델만 로드 (CPU 환경)"""

    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python이 설치되지 않았습니다.")
        print("설치: pip install llama-cpp-python")
        return None, None

    # GGUF 모델 경로
    gguf_model_path = "./merged_model-finetuned-q5_k_m.gguf"

    if not os.path.exists(gguf_model_path):
        print(f"GGUF 모델 파일을 찾을 수 없습니다: {gguf_model_path}")
        return None, None

    print(f"GGUF 모델 로딩: {gguf_model_path}")

    try:
        # CPU만 사용하는 설정
        model = Llama(
            model_path=gguf_model_path,
            n_ctx=2048,  # 컨텍스트 길이
            n_gpu_layers=0,  # CPU만 사용 (GPU 없음)
            verbose=False,
            n_threads=4  # CPU 스레드 수
        )

        # 토크나이저는 원본 모델에서 로드
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/Phi-3-mini-128k-instruct",
                trust_remote_code=True
            )
        except:
            print("원본 토크나이저 로딩 실패, 기본 토크나이저 사용")
            tokenizer = None

        print("GGUF 모델 로딩 완료!")
        return model, tokenizer

    except Exception as e:
        print(f"첫 번째 시도 실패: {e}")
        print("더 작은 설정으로 재시도...")

        # 더욱 극단적인 메모리 절약 시도
        try:
            model = Llama(
                model_path=gguf_model_path,
                n_ctx=256,  # 매우 작은 컨텍스트
                n_gpu_layers=0,
                verbose=True,
                n_threads=1,  # 단일 스레드
                use_mmap=True,
                use_mlock=False,
                n_batch=1,  # 최소 배치
                low_vram=True,
                f16_kv=True
            )
            print("극소 설정으로 모델 로딩 성공!")
        except Exception as e2:
            print(f"극소 설정도 실패: {e2}")
            print("\n=== 문제 해결 방법 ===")
            print("1. 메모리 부족: 다른 프로그램들을 종료해보세요")
            print("2. 모델 파일 손상: GGUF 파일을 다시 다운로드하세요")
            print("3. llama-cpp-python 재설치:")
            print("   pip uninstall llama-cpp-python")
            print("   pip install llama-cpp-python --no-cache-dir")
            return None, None


def format_phi3_prompt(prompt):
    """Phi-3 프롬프트 포맷팅"""
    return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"


def generate_response(model, prompt, max_tokens=512, temperature=0.7):
    """GGUF 모델로 텍스트 생성"""

    if model is None:
        return "모델이 로드되지 않았습니다."

    # Phi-3 포맷 적용
    formatted_prompt = format_phi3_prompt(prompt)

    try:
        response = model.create_completion(
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            echo=False,
            stop=["<|end|>", "<|user|>", "\n사용자:", "\nUser:"]
        )

        result = response['choices'][0]['text'].strip()
        # 불필요한 토큰 제거
        result = result.replace("<|end|>", "").strip()
        return result

    except Exception as e:
        return f"생성 오류: {e}"


def interactive_chat():
    """대화형 채팅"""
    print("모델 로딩 중...")
    model, tokenizer = load_gguf_model()

    if model is None:
        print("모델 로딩에 실패했습니다.")
        return

    print("채팅을 시작합니다! (종료: quit, exit, q)")
    print("-" * 50)

    while True:
        user_input = input("\n사용자: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("채팅을 종료합니다.")
            break

        if not user_input:
            continue

        print("AI 생각 중...")
        response = generate_response(model, user_input)
        print(f"AI: {response}")


def test_samples():
    """샘플 테스트"""
    print("모델 로딩 중...")
    model, tokenizer = load_gguf_model()

    if model is None:
        print("모델 로딩에 실패했습니다.")
        return

    print("샘플 테스트 시작!")
    print("=" * 50)

    test_prompts = [
        "안녕하세요! 자기소개를 해주세요.",
        "Python으로 'Hello World'를 출력하는 코드를 작성해주세요.",
        "인공지능이 무엇인지 간단히 설명해주세요.",
        "오늘 기분이 어떠세요?",
        "간단한 수학 문제: 15 + 27은?"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n=== 테스트 {i} ===")
        print(f"질문: {prompt}")

        response = generate_response(model, prompt, max_tokens=256)
        print(f"답변: {response}")
        print("-" * 30)


def main():
    """메인 함수"""
    print("GGUF 모델 실행기 (CPU 전용)")
    print("=" * 40)
    print("모델: merged_model-finetuned-q5_k_m.gguf")
    print("환경: CPU 전용")

    print("\n실행 모드를 선택하세요:")
    print("1. 대화형 채팅")
    print("2. 샘플 테스트")

    choice = input("선택 (1 또는 2): ").strip()

    if choice == "1":
        interactive_chat()
    elif choice == "2":
        test_samples()
    else:
        print("잘못된 선택입니다. 샘플 테스트를 실행합니다.")
        test_samples()


if __name__ == "__main__":
    main()