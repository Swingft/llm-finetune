import os
import shutil
import warnings
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# =========================
# 0) 환경 / 토큰
# =========================
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")

# 캐시/경고 소음 줄이기
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings("ignore", category=UserWarning)

# PyTorch matmul 정밀도(속도/안정성 밸런스)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# =========================
# 1) 하이퍼파라미터
# =========================
EPOCHS = 3
LR = 2e-4
TRAIN_BS = 1
GRAD_ACCUM = 16
EVAL_BS = 1
PACKING = True
LOG_STEPS = 10
EVAL_STEPS = 50
SAVE_STEPS = 100
SEED = 42
MAX_LENGTH = None  # 필요 시 4096 등

USE_LORA = True
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.05
LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# =========================
# 2) 경로/모델
# =========================
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

TRAIN_JSONL = "train.jsonl"
DATASET_NAME = "MyDataset"  # 데이터셋 특성에 맞게 변경

EVAL_JSONL = ""  # 선택 (없으면 빈 문자열)
OUTPUT_DIR = f"./out/{MODEL_ID.split('/')[-1]}_lora_adapter_r{LORA_R}_{DATASET_NAME}"

# 디바이스/정밀도
DEVICE_MAP = "auto"
USE_BF16 = torch.cuda.is_available() and DEVICE_MAP != "cpu"

# =========================
# 3) 허깅페이스 로그인
# =========================
if HF_TOKEN:
    try:
        from huggingface_hub import login as hf_login

        hf_login(token=HF_TOKEN.strip())
        print("✅ HF 로그인 완료")
    except Exception as e:
        print(f"⚠️ HF 로그인 실패(무시): {e}")


# =========================
# 4) 데이터 로더
# =========================
def load_and_prepare_jsonl(train_path: str, eval_path: Optional[str], text_field: str = "text"):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"학습 파일 없음: {train_path}")

    data_files = {"train": train_path}
    if eval_path:
        if not os.path.exists(eval_path):
            raise FileNotFoundError(f"평가 파일 없음: {eval_path}")
        data_files["validation"] = eval_path

    ds = load_dataset("json", data_files=data_files)
    tr = ds["train"];
    ev = ds.get("validation")

    cols = set(tr.column_names)
    if not ({"instruction", "output"}.issubset(cols) or {"input", "output"}.issubset(cols)):
        raise ValueError(f"데이터 컬럼이 맞지 않습니다. 실제 컬럼: {tr.column_names}")

    def to_text(ex):
        instr = (ex.get("instruction") or ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()
        inp = (ex.get("input") if "instruction" in ex else "").strip()

        if not instr and inp and "instruction" not in ex:
            instr, inp = inp, ""

        if not instr or not out: return None

        if inp:
            return {text_field: f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"}
        else:
            return {text_field: f"### Instruction:\n{instr}\n\n### Response:\n{out}"}

    tr = tr.map(to_text, remove_columns=list(cols)).filter(lambda x: x is not None)
    if ev:
        ev = ev.map(to_text, remove_columns=list(cols)).filter(lambda x: x is not None)

    return tr, ev


# =========================
# 5) 메인 함수
# =========================
def main():
    # 모델 접근 권한 확인
    try:
        from huggingface_hub import HfApi
        HfApi().model_info(MODEL_ID, token=HF_TOKEN)
    except Exception as e:
        raise RuntimeError(f"모델 리포 확인 실패: {MODEL_ID}\n원인: {e}")

    # 시드 고정
    try:
        import random, numpy as np
        random.seed(SEED);
        np.random.seed(SEED);
        torch.manual_seed(SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    except Exception:
        pass

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 50 + "\n🚀 LoRA 어댑터 학습 시작\n" + "=" * 50)
    print(f"Model ID: {MODEL_ID}\nOutput Dir: {OUTPUT_DIR}\nLoRA r: {LORA_R}")
    print("=" * 50 + "\n")

    # 토크나이저 로드
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True, token=HF_TOKEN)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=DEVICE_MAP,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float32,
        trust_remote_code=True,
        token=HF_TOKEN,
        attn_implementation="eager"
    )

    if torch.cuda.is_available(): model.gradient_checkpointing_enable()
    print("✅ 모델/토크나이저 로드 완료")

    # 데이터 로드
    train_ds, eval_ds = load_and_prepare_jsonl(TRAIN_JSONL, EVAL_JSONL or None)
    print(f"✅ 데이터 준비 완료 (학습: {len(train_ds)}개)")

    # SFT 설정
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        packing=PACKING,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=LOG_STEPS,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=EVAL_STEPS if eval_ds else None,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        seed=SEED,
        bf16=USE_BF16,
        fp16=False,
        report_to=[],
        gradient_checkpointing=True,
        max_seq_length=MAX_LENGTH if MAX_LENGTH else 2048
    )

    # LoRA 설정
    from peft import LoraConfig, TaskType
    peft_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[t.strip() for t in LORA_TARGETS.split(",")]
    )

    # 트레이너 초기화 및 학습
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        peft_config=peft_cfg,
    )

    print("🚀 학습 시작!")
    trainer.train()
    print("✅ 학습 완료!")

    # 최종 어댑터 저장
    print("💾 최종 LoRA 어댑터 저장 중...")
    trainer.save_model(OUTPUT_DIR)
    print(f"✅ LoRA 어댑터 저장 완료: {OUTPUT_DIR}")

    print("\n" + "=" * 60 + "\n🎉 LoRA 어댑터 학습 완료!\n" + "=" * 60)
    print(f"📁 최종 어댑터 경로: {OUTPUT_DIR}")
    print("\n다음 단계:")
    print("1. (선택) 이 어댑터를 베이스 모델과 병합합니다.")
    print("2. (선택) 이 어댑터를 GGUF로 변환하여 베이스 GGUF와 함께 사용합니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
