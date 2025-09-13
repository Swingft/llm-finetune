from llama_cpp import Llama
import time

MODEL_PATH = "merged_model-finetuned-q5_k_m.gguf"

print("Loading model for Apple Metal...")
start_time = time.time()

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=0,  # -1로 설정 시, 가능한 모든 레이어를 Metal GPU로 오프로딩
    n_ctx=4096,
    verbose=True
)

end_time = time.time()
print(f"Model loaded in {end_time - start_time:.2f} seconds.")
print("-" * 50)

# --- 2. 프롬프트 준비 ---
# 파인튜닝 시 사용했던 프롬프트 템플릿에 맞춰 질문을 구성해야 모델이 가장 잘 응답합니다.
# (예: Phi-3 템플릿)
user_question = "딥러닝과 머신러닝의 차이점을 설명해줘."

prompt = f"""<|system|>
You are a helpful AI assistant.<|end|>
<|user|>
{user_question}<|end|>
<|assistant|>"""


# --- 3. 텍스트 생성 (추론) ---
print("Generating response...")
start_time = time.time()

output = llm(
    prompt,
    max_tokens=1024,      # 최대 생성 토큰 수
    temperature=0.7,      # 생성의 다양성을 조절 (0에 가까울수록 결정적)
    top_p=0.9,            # 생성 토큰의 후보군을 제한
    stop=["<|end|>"],     # 응답 생성을 멈출 특정 토큰 지정
    echo=False            # True로 설정 시, 입력한 프롬프트까지 함께 출력
)

end_time = time.time()
print(f"Response generated in {end_time - start_time:.2f} seconds.")
print("-" * 50)


# --- 4. 결과 출력 ---
generated_text = output['choices'][0]['text'].strip()
print("🤖 모델 응답:\n")
print(generated_text)