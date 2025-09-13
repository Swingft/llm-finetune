import os
import shutil
import warnings
from typing import Optional
import psutil
import time
import threading

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import PeftModel

# =========================
# 0) 환경 / 토큰
# =========================
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings("ignore", category=UserWarning)
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
MAX_LENGTH = None

USE_LORA = True
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.05
LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# =========================
# 2) 경로/모델
# =========================
MODEL_ID = "microsoft/Phi-3-mini-128k-instruct"
TRAIN_JSONL = "json_data.jsonl"
DATASET_NAME = "sensitive"

EVAL_JSONL = ""
OUTPUT_DIR = f"./out/{MODEL_ID.split('/')[-1]}_merged_r{LORA_R}_{DATASET_NAME}"
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
# 4) 유틸리티 함수 (데이터 로더, 압축 등)
# =========================
def load_and_prepare_jsonl(train_path: str, eval_path: Optional[str], text_field: str = "text"):
    if not os.path.exists(train_path): raise FileNotFoundError(f"학습 파일 없음: {train_path}")
    data_files = {"train": train_path}
    if eval_path:
        if not os.path.exists(eval_path): raise FileNotFoundError(f"평가 파일 없음: {eval_path}")
        data_files["validation"] = eval_path
    ds = load_dataset("json", data_files=data_files)
    tr, ev = ds["train"], ds.get("validation")
    cols = set(tr.column_names)
    if not ({"instruction", "output"}.issubset(cols) or {"input", "output"}.issubset(cols)):
        raise ValueError(f"데이터 컬럼이 맞지 않습니다. 실제 컬럼: {tr.column_names}")

    def to_text(ex):
        instr = (ex.get("instruction") or ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()
        inp = (ex.get("input") if "instruction" in ex else "").strip()
        if not instr and inp and "instruction" not in ex: instr, inp = inp, ""
        if not instr or not out: return None
        prompt = f"### Instruction:\n{instr}\n\n" + (f"### Input:\n{inp}\n\n" if inp else "") + f"### Response:\n{out}"
        return {text_field: prompt}

    tr = tr.map(to_text, remove_columns=list(cols)).filter(lambda x: x is not None)
    if ev: ev = ev.map(to_text, remove_columns=list(cols)).filter(lambda x: x is not None)
    return tr, ev


def format_bytes(b):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if b < 1024: return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} {unit}"


def safe_compress_directory(source_dir, archive_name, max_wait_time=900):
    if not os.path.exists(source_dir): return False
    print("📦 결과물 압축 중...")
    try:
        shutil.make_archive(archive_name, "zip", source_dir)
        print(f"✅ 압축 완료: {archive_name}.zip")
        return True
    except Exception as e:
        print(f"❌ 압축 실패: {e}")
        return False


# =========================
# 5) 메인 함수
# =========================
def main():
    try:
        from huggingface_hub import HfApi
        HfApi().model_info(MODEL_ID, token=HF_TOKEN)
    except Exception as e:
        raise RuntimeError(f"모델 리포 확인 실패: {MODEL_ID}\n원인: {e}")

    try:
        import random, numpy as np
        random.seed(SEED);
        np.random.seed(SEED);
        torch.manual_seed(SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    except Exception:
        pass

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 50 + "\n🚀 LoRA 학습 및 병합 시작\n" + "=" * 50)
    print(f"Model ID: {MODEL_ID}\nOutput Dir: {OUTPUT_DIR}")
    print("=" * 50 + "\n")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True, token=HF_TOKEN)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=DEVICE_MAP,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float32,
        trust_remote_code=True,
        token=HF_TOKEN,
        attn_implementation="eager"
    )
    if torch.cuda.is_available(): model.gradient_checkpointing_enable()

    train_ds, eval_ds = load_and_prepare_jsonl(TRAIN_JSONL, EVAL_JSONL or None)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        packing=PACKING,
        per_device_train_batch_size=TRAIN_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=LOG_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        seed=SEED,
        bf16=USE_BF16,
        report_to=[],
        max_seq_length=MAX_LENGTH if MAX_LENGTH else 2048
    )

    from peft import LoraConfig, TaskType
    peft_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[t.strip() for t in LORA_TARGETS.split(",")]
    )

    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_ds, tokenizer=tok, peft_config=peft_cfg)

    print("🚀 학습 시작!")
    trainer.train()
    print("✅ 학습 완료!")

    print("🔄 LoRA 어댑터를 베이스 모델과 병합 및 저장 중...")
    merged_model = trainer.model.merge_and_unload()
    merged_output_dir = os.path.join(OUTPUT_DIR, "merged_model")
    merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
    tok.save_pretrained(merged_output_dir)
    print(f"✅ 병합된 모델 저장 완료: {merged_output_dir}")

    safe_compress_directory(OUTPUT_DIR, f"{OUTPUT_DIR}_complete")

    print("\n" + "=" * 60 + "\n🎉 파인튜닝 및 병합 완료!\n" + "=" * 60)
    print(f"📁 최종 모델 경로: {merged_output_dir}")
    print(f"🔄 GGUF 변환용 경로: {merged_output_dir}")
    print("\n다음 단계:\n1. GGUF 변환 스크립트 실행\n2. 생성된 GGUF 파일을 맥북으로 다운로드")
    print("=" * 60)


if __name__ == "__main__":
    main()
