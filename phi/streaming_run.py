#!/usr/bin/env python3
import gc
import os


def load_minimal_model():
    """최소 메모리로 모델 로드"""

    # 메모리 정리
    gc.collect()

    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python 설치 필요: pip install llama-cpp-python")
        return None

    gguf_path = "./merged_model-finetuned-q5_k_m.gguf"

    print("초절약 모드로 모델 로딩...")
    print("(첫 로딩은 시간이 오래 걸릴 수 있습니다)")

    try:
        model = Llama(
            model_path=gguf_path,

            # 메모리 최소화 설정
            n_ctx=128,  # 매우 짧은 컨텍스트
            n_gpu_layers=0,  # CPU만
            n_threads=1,  # 단일 스레드
            n_batch=1,  # 최소 배치

            # 메모리 맵핑 최적화
            use_mmap=True,  # 메모리 맵 사용
            use_mlock=False,  # 메모리 락 비활성화

            # 양자화 최적화
            f16_kv=True,  # KV 캐시 16비트
            low_vram=True,  # 저메모리 모드

            # 디버그
            verbose=False,

            # 추가 절약 옵션
            logits_all=False,  # 모든 토큰의 로짓 저장 안함
            vocab_only=False,  # 어휘만 로드하지 않음
            embedding=False  # 임베딩 비활성화
        )

        print("✅ 초절약 모드 로딩 성공!")
        return model

    except Exception as e:
        print(f"로딩 실패: {e}")

        # 마지막 수단: 더욱 극단적 설정
        try:
            print("마지막 수단: 극한 절약 모드...")
            model = Llama(
                model_path=gguf_path,
                n_ctx=64,  # 극소 컨텍스트
                n_gpu_layers=0,
                n_threads=1,
                n_batch=1,
                use_mmap=True,
                use_mlock=False,
                f16_kv=True,
                low_vram=True,
                verbose=False
            )
            print("✅ 극한 절약 모드 성공!")
            return model

        except Exception as e2:
            print(f"극한 모드도 실패: {e2}")
            return None


def generate_short_response(model, prompt, max_tokens=50):
    """짧은 응답 생성 (메모리 절약)"""

    if model is None:
        return "모델 로드 실패"

    # 간단한 프롬프트 (Phi-3 형식 생략해서 메모리 절약)
    try:
        response = model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,  # 짧은 응답
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stop=["\n", ".", "!", "?"]  # 빨리 중단
        )

        result = response['choices'][0]['text'].strip()

        # 메모리 정리
        gc.collect()

        return result

    except Exception as e:
        return f"생성 실패: {e}"


def chat_minimal():
    """최소 메모리 채팅"""

    print("=== 초절약 모드 채팅 ===")
    print("- 짧은 답변만 가능")
    print("- 한 번에 하나씩만 처리")
    print("- 종료: quit")
    print("-" * 30)

    model = load_minimal_model()

    if model is None:
        print("모델 로딩 실패. 메모리를 더 확보해주세요.")
        return

    while True:
        user_input = input("\n사용자: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("채팅 종료")
            break

        if not user_input:
            continue

        print("생각 중... (잠시만요)")

        # 짧은 응답만
        response = generate_short_response(model, user_input, max_tokens=30)
        print(f"AI: {response}")

        # 매번 메모리 정리
        gc.collect()


def main():
    print("메모리 부족 환경용 초절약 실행기")
    print("=" * 40)

    # 현재 메모리 체크
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / 1024 ** 3
        print(f"사용 가능 메모리: {available_gb:.1f} GB")

        if available_gb < 2:
            print("⚠️  메모리가 매우 부족합니다!")
            print("다른 프로그램들을 종료하고 다시 시도해보세요.")

            choice = input("그래도 시도하시겠습니까? (y/N): ")
            if choice.lower() != 'y':
                return
    except:
        pass

    chat_minimal()


if __name__ == "__main__":
    main()