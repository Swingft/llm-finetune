#!/usr/bin/env python3
"""
데이터 타입 문제를 해결한 견고한 학습 스크립트
"""

import os
import warnings
import json

warnings.filterwarnings("ignore")

# 환경 설정
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from datasets import load_dataset
    from peft import LoraConfig, TaskType

    # TRL import 시도
    try:
        from trl import SFTTrainer, SFTConfig

        print("TRL SFTConfig 사용")
        USE_SFT_CONFIG = True
    except ImportError:
        try:
            from trl import SFTTrainer

            SFTConfig = TrainingArguments
            print("TRL SFTTrainer + TrainingArguments 사용")
            USE_SFT_CONFIG = False
        except ImportError:
            print("TRL import 실패")
            exit(1)

    print("모든 라이브러리 로드 완료")

except ImportError as e:
    print(f"라이브러리 로드 실패: {e}")
    exit(1)

# 설정값
MODEL_ID = "microsoft/Phi-3-mini-128k-instruct"
TRAIN_JSONL = "./exclude.jsonl"
OUTPUT_DIR = "./out/Phi-3-mini-128k-instruct_merged_r128_exclude"


def safe_str_convert(value):
    """안전하게 값을 문자열로 변환"""
    if value is None:
        return ""
    elif isinstance(value, str):
        return value.strip()
    elif isinstance(value, dict):
        # dict인 경우 JSON 문자열로 변환
        return json.dumps(value, ensure_ascii=False, indent=2)
    elif isinstance(value, list):
        # list인 경우 적절히 변환
        return ", ".join(str(item) for item in value)
    else:
        return str(value).strip()


def check_training_completed():
    """학습 완료 확인"""
    merged_path = os.path.join(OUTPUT_DIR, "merged_model")
    config_exists = os.path.exists(os.path.join(merged_path, "config.json"))
    tokenizer_exists = os.path.exists(os.path.join(merged_path, "tokenizer.json"))

    weight_files = [
        os.path.join(merged_path, "pytorch_model.bin"),
        os.path.join(merged_path, "model.safetensors"),
        os.path.join(merged_path, "model-00001-of-00002.safetensors")
    ]
    weight_exists = any(os.path.exists(f) for f in weight_files)

    return config_exists and tokenizer_exists and weight_exists


def prepare_dataset():
    """데이터셋 준비 - 타입 안전성 보장"""
    if not os.path.exists(TRAIN_JSONL):
        raise FileNotFoundError(f"학습 파일이 없습니다: {TRAIN_JSONL}")

    print("데이터셋 로드 중...")
    dataset = load_dataset("json", data_files={"train": TRAIN_JSONL})
    train_ds = dataset["train"]

    print(f"원본 데이터: {len(train_ds)}개")

    # 첫 번째 샘플 확인
    if len(train_ds) > 0:
        sample = train_ds[0]
        print("샘플 데이터 구조:")
        for key, value in sample.items():
            print(f"  {key}: {type(value)} - {str(value)[:100]}...")

    def format_data(example):
        """데이터 형식 변환 - 타입 안전성 보장"""
        try:
            # instruction 필드 처리
            instruction = safe_str_convert(example.get("instruction") or example.get("input"))

            # output 필드 처리
            output_raw = example.get("output")

            if isinstance(output_raw, dict):
                # dict인 경우 특별 처리
                if "reasoning" in output_raw and "identifiers" in output_raw:
                    # Swift 코드 분석 결과 형식
                    reasoning = safe_str_convert(output_raw.get("reasoning", ""))
                    identifiers = output_raw.get("identifiers", [])

                    if isinstance(identifiers, list):
                        identifiers_str = ", ".join(str(item) for item in identifiers)
                    else:
                        identifiers_str = safe_str_convert(identifiers)

                    output = f"Reasoning: {reasoning}\n\nIdentifiers: {identifiers_str}"
                else:
                    # 일반 dict인 경우 JSON으로 변환
                    output = json.dumps(output_raw, ensure_ascii=False, indent=2)
            else:
                output = safe_str_convert(output_raw)

            # input 필드 처리 (있는 경우)
            input_text = safe_str_convert(
                example.get("input")) if "input" in example and "instruction" in example else ""

            # 최종 검증
            if not instruction or not output:
                print(f"빈 필드 발견 - instruction: '{instruction[:50]}...', output: '{output[:50]}...'")
                return {"text": None}  # 필터링될 예정

            # 프롬프트 생성
            if input_text:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

            return {"text": text}

        except Exception as e:
            print(f"데이터 처리 에러: {e}")
            print(f"문제 데이터: {example}")
            return {"text": None}  # 필터링될 예정

    # 데이터 변환
    print("데이터 변환 중...")
    train_ds = train_ds.map(format_data, remove_columns=train_ds.column_names)

    # None 값 필터링
    print("빈 데이터 필터링 중...")
    train_ds = train_ds.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)

    print(f"최종 학습 데이터: {len(train_ds)}개")

    if len(train_ds) == 0:
        raise ValueError("사용 가능한 학습 데이터가 없습니다!")

    # 변환된 샘플 확인
    print("변환된 데이터 샘플:")
    print(train_ds[0]["text"][:200] + "...")

    return train_ds


def main():
    if check_training_completed():
        print("학습이 이미 완료되어 있습니다!")
        print(f"경로: {os.path.join(OUTPUT_DIR, 'merged_model')}")
        return

    print("학습 시작...")

    # HF 로그인
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        print("HuggingFace 로그인 완료")
    except Exception as e:
        print(f"HF 로그인 실패: {e}")

    # 토크나이저 로드
    print("토크나이저 로드...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드
    print("모델 로드...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
        trust_remote_code=True,
        attn_implementation="eager"
    )

    # 데이터셋 준비
    try:
        train_dataset = prepare_dataset()
    except Exception as e:
        print(f"데이터셋 준비 실패: {e}")
        print("데이터 형식 확인이 필요합니다.")
        return

    # LoRA 설정
    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 학습 설정
    if USE_SFT_CONFIG:
        training_args = SFTConfig(
            output_dir=OUTPUT_DIR,
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            bf16=True,
            report_to=[],
            packing=True
        )
    else:
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            bf16=True,
            report_to=[],
            remove_unused_columns=False
        )

    # 트레이너 생성
    if USE_SFT_CONFIG:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            dataset_text_field="text",
            packing=True,
            max_seq_length=2048
        )

    # 학습 실행
    print("학습 시작!")
    trainer.train()
    print("학습 완료!")

    # 모델 병합 및 저장
    print("모델 병합...")
    merged_model = trainer.model.merge_and_unload()
    merged_path = os.path.join(OUTPUT_DIR, "merged_model")

    merged_model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)

    print(f"병합 완료: {merged_path}")
    print("모든 작업 완료!")


if __name__ == "__main__":
    main()