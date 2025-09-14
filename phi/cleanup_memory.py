#!/usr/bin/env python3
import psutil
import gc
import os


def cleanup_memory():
    """메모리 정리"""
    print("=== 메모리 정리 시작 ===")

    # 1. Python 가비지 컬렉션
    print("1. Python 가비지 컬렉션...")
    gc.collect()

    # 2. 메모리 사용량이 큰 프로세스 찾기
    print("2. 메모리 많이 쓰는 프로세스:")
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            processes.append({
                'pid': proc.info['pid'],
                'name': proc.info['name'],
                'memory_mb': proc.info['memory_info'].rss / 1024 / 1024
            })
        except:
            continue

    # 메모리 사용량 순 정렬
    processes.sort(key=lambda x: x['memory_mb'], reverse=True)

    print("상위 10개 프로세스:")
    for i, proc in enumerate(processes[:10]):
        print(f"{i + 1:2d}. {proc['name']:20s} {proc['memory_mb']:6.0f}MB (PID: {proc['pid']})")

    # 3. 메모리 상태 출력
    memory = psutil.virtual_memory()
    print(f"\n현재 메모리 사용률: {memory.percent:.1f}%")
    print(f"사용 가능: {memory.available / 1024 ** 3:.1f} GB")

    return memory.available / 1024 ** 3


def suggest_cleanup():
    """정리 방법 제안"""
    print("\n=== 메모리 확보 방법 ===")
    print("1. 브라우저 탭 많이 열었다면 닫기")
    print("2. 사용하지 않는 애플리케이션 종료")
    print("3. PyCharm/IDE 메모리 설정 낮추기")
    print("4. 터미널에서 실행:")
    print("   sudo purge  # macOS 메모리 정리")
    print("   # 또는")
    print("   echo 3 | sudo tee /proc/sys/vm/drop_caches  # Linux")

    print("\n=== 대안 방법 ===")
    print("A. 더 작은 양자화 버전 사용 (Q3_K_M)")
    print("B. 스트리밍 모드로 실행")
    print("C. 가상메모리 증가 (스왑)")


if __name__ == "__main__":
    available = cleanup_memory()
    suggest_cleanup()

    if available >= 4:
        print(f"\n✅ {available:.1f}GB 사용 가능 - 모델 로딩 시도해볼만함!")
    else:
        print(f"\n❌ {available:.1f}GB로 부족 - 더 정리하거나 대안 방법 필요")