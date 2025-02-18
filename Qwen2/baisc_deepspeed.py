#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
이 스크립트는:
1. Qwen2AudioForConditionalGeneration 모델을 4-bit 양자화로 로드
2. DeepSpeed(ZeRO-3 + CPU offload)로 초기화
3. JSON 파일을 읽고 CSV로 결과를 저장 (기존 코드 유지)
   - AAC_JSON, ASR_JSON → 결과 AAC_CSV, ASR_CSV
"""

import os
import json
import csv
import shutil
import torch
import deepspeed
from tqdm import tqdm
from scipy.signal import resample_poly
import soundfile as sf

# transformers
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
)
# Qwen2-Audio
from transformers import Qwen2AudioForConditionalGeneration

# DeepSpeed + HF 통합
from transformers.integrations import HfDeepSpeedConfig

# ------------------------------------------------------------------------
# 0) 분산 환경 / DeepSpeed 초기 설정
# ------------------------------------------------------------------------
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

# GPU 할당
torch.cuda.set_device(local_rank)
# 분산 초기화 (DeepSpeed)
deepspeed.init_distributed()

# ------------------------------------------------------------------------
# 1) 사용자 환경 상수 (기존 코드 유지)
# ------------------------------------------------------------------------
RESULT_AAC_CSV = "results/qwen_aac.csv"
RESULT_ASR_CSV = "results/qwen_asr.csv"
AAC_JSON = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/json/test_aac.json"
ASR_JSON = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/json/test_asr.json"
CHECKPOINT = "Qwen/Qwen2-Audio-7B"
PREFIX = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/data"

# ------------------------------------------------------------------------
# 2) DeepSpeed 설정(JSON 대신 파이썬 dict로 정의) - huggingface 예시 기반
# ------------------------------------------------------------------------
ds_config = {
    "fp16": {
        "enabled": False       # Ampere 이상 GPU 시 권장
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
        "device": "none"  
        },
        "offload_param": {
        "device": "none"  
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5.0e7,  
        "stage3_prefetch_bucket_size": 5.0e7,
        "stage3_param_persistence_threshold": 1e5
    },
    "steps_per_print": 2000,
    "train_batch_size": 2,                  # world_size=2, micro_batch_size=1 -> grad_acc_steps=1
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "wall_clock_breakdown": False
}

# DeepSpeed + HF 통합 설정
ds_hf = HfDeepSpeedConfig(ds_config)

# ------------------------------------------------------------------------
# 3) 4-bit 양자화로 모델 로드 (Pruning 없음)
# ------------------------------------------------------------------------
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # fp16 계산
)

print(f"[Rank {local_rank}] Loading Qwen2Audio model in 4-bit quantization...")

model_4bit = Qwen2AudioForConditionalGeneration.from_pretrained(
    CHECKPOINT,
    quantization_config=bnb_config_4bit,
    trust_remote_code=True,
    device_map=None,    # DeepSpeed가 분산 처리
)

# Processor 로드
processor = None
if local_rank == 0:
    print("[Rank 0] Loading processor...")
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

# 모든 rank 동기화
torch.distributed.barrier()
if processor is None:
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

# ------------------------------------------------------------------------
# 4) DeepSpeed ZeRO-3 초기화
# ------------------------------------------------------------------------
ds_engine = deepspeed.initialize(
    model=model_4bit,
    config_params=ds_config
)[0]
ds_engine.module.eval()

# ------------------------------------------------------------------------
# 5) JSON 처리 및 Inference 함수 (기존 코드 유지)
# ------------------------------------------------------------------------
def process_audio_file(file_path, prompt, max_duration=10):
    audio, sr = sf.read(file_path)
    if sr != 16000:
        audio = resample_poly(audio, up=16000, down=sr)
        sr = 16000

    audio = audio[: sr * max_duration]  # Truncate audio

    # local_rank 기준으로 이동
    inputs = processor(text=[prompt], audios=audio, return_tensors="pt").to(local_rank)

    with torch.inference_mode():
        generated_ids = ds_engine.module.generate(**inputs, max_length=256, use_cache=False)
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    return response

def process_json(json_path, result_csv, batch_size=1):
    if local_rank == 0:
        print(f"[Rank 0] Processing JSON file: {json_path}")
    # rank=0만 실제 파일 write (원본 로직)
    if local_rank == 0:
        with open(json_path, 'r') as f:
            data = json.load(f)

        with open(result_csv, 'w', newline='') as csvfile:
            fieldnames = ['testset_id', 'text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            annotations = data['annotation']
            for i in tqdm(range(0, len(annotations), batch_size), desc=f"Processing {json_path}"):
                batch = annotations[i:i + batch_size]
                for item in batch:
                    testset_id = item['testset_id']
                    audio_path = os.path.join(PREFIX, item['path'])
                    prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
                    try:
                        transcription = process_audio_file(audio_path, prompt)
                        writer.writerow({'testset_id': testset_id, 'text': transcription})
                    except RuntimeError as e:
                        print(f"Error processing {audio_path}: {e}")
                        torch.cuda.empty_cache()

    # 모든 rank 동기화 (rank=0이 작업 완료할 때까지 대기)
    torch.distributed.barrier()

# ------------------------------------------------------------------------
# 6) 실제 Inference 수행
# ------------------------------------------------------------------------
if local_rank == 0:
    print("[Rank 0] Starting inference on AAC and ASR JSON...")
process_json(AAC_JSON, RESULT_AAC_CSV)
process_json(ASR_JSON, RESULT_ASR_CSV)

# ------------------------------------------------------------------------
# 7) 마무리
# ------------------------------------------------------------------------
if local_rank == 0:
    print("[Rank 0] Processing completed.")
