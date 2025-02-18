#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSDP 기반 Qwen2Audio 추론 코드
"""

import os
import json
import csv
from tqdm import tqdm
import soundfile as sf
from scipy.signal import resample_poly

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.utils import prune

from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
    BitsAndBytesConfig,
)

# ------------------------------------------------------------------------
# 1) 사용자 환경 설정
# ------------------------------------------------------------------------
RESULT_AAC_CSV = "results/qwen_aac.csv"
RESULT_ASR_CSV = "results/qwen_asr.csv"
AAC_JSON = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/json/test_aac.json"
ASR_JSON = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/json/test_asr.json"
CHECKPOINT = "Qwen/Qwen2-Audio-7B"
PREFIX = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/data"

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

# ------------------------------------------------------------------------
# 2) PyTorch FSDP 초기화
# ------------------------------------------------------------------------
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)

# ------------------------------------------------------------------------
# 3) BitsAndBytes 4-bit 양자화 설정
# ------------------------------------------------------------------------
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # FP16 계산
)

# ------------------------------------------------------------------------
# 4) FSDP 관련 정책 및 옵션 설정
# ------------------------------------------------------------------------
# MixedPrecision: 모든 연산을 FP16으로 설정
mp_policy = MixedPrecision(
    param_dtype=torch.float16,  # 모델 파라미터를 FP16으로 유지
    reduce_dtype=torch.float16,  # 통신 시 데이터 타입
    buffer_dtype=torch.float16,  # 모델 버퍼도 FP16
)

# Auto-wrap 정책 생성 함수
def get_auto_wrap_policy(min_params=1e7):
    """
    FSDP auto_wrap_policy 설정.
    min_params: 특정 모듈의 파라미터 개수가 이 값 이상일 때만 래핑.
    """
    return lambda module, recurse, nonwrapped_numel: size_based_auto_wrap_policy(
        module=module,
        recurse=recurse,
        nonwrapped_numel=nonwrapped_numel,
        min_num_params=min_params,
    )

# ------------------------------------------------------------------------
# 5) 모델 로드 및 Pruning 적용
# ------------------------------------------------------------------------
if local_rank == 0:
    print(f"[Rank {local_rank}] Loading model...")

# 모델 설정 로드
config = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True)
config.torch_dtype = torch.float16

# 모델 로드 (4-bit 양자화 포함)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    CHECKPOINT,
    config=config,
    quantization_config=bnb_config_4bit,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# Pruning 적용: 모든 Linear 레이어에서 50% 파라미터 제거
def apply_pruning(model, amount=0.5):
    """
    Prune 50% of weights in Linear layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")  # Pruning mask 제거
    return model

model = apply_pruning(model, amount=0.5)

# 모든 파라미터의 requires_grad를 False로 설정
for param in model.parameters():
    param.requires_grad = False

# FSDP 래핑
ignored_modules = [module for module in model.modules() if isinstance(module, torch.nn.Embedding)]

model = FSDP(
    model,
    auto_wrap_policy=get_auto_wrap_policy(min_params=1e7),
    mixed_precision=mp_policy,
    device_id=torch.cuda.current_device(),
    ignored_modules=ignored_modules,  # 플래트닝 제외 모듈
    use_orig_params=True,  # 플래트닝 문제 완화
)

model.eval()

# Processor (Tokenizer 등) 로드
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

# 모든 rank 동기화
dist.barrier()

# ------------------------------------------------------------------------
# 6) JSON 처리 및 추론 함수
# ------------------------------------------------------------------------
def process_audio_file(file_path, prompt, max_duration=10):
    audio, sr = sf.read(file_path)
    if sr != 16000:
        audio = resample_poly(audio, up=16000, down=sr)
        sr = 16000

    audio = audio[: sr * max_duration]  # Truncate audio

    inputs = processor(text=[prompt], audios=audio, return_tensors="pt")
    inputs = {k: v.to(local_rank) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_length=256, use_cache=False)
    generated_ids = generated_ids[:, inputs["input_ids"].size(1):]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    return response

def process_json(json_path, result_csv, batch_size=1):
    if local_rank == 0:
        print(f"[Rank {local_rank}] Processing JSON file: {json_path}")
    # rank=0에서만 결과 CSV 생성
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

    # 모든 rank 동기화
    dist.barrier()

# ------------------------------------------------------------------------
# 7) Inference 수행
# ------------------------------------------------------------------------
if local_rank == 0:
    print("[Rank 0] Starting inference on AAC and ASR JSON...")

process_json(AAC_JSON, RESULT_AAC_CSV)
process_json(ASR_JSON, RESULT_ASR_CSV)

if local_rank == 0:
    print("[Rank 0] Processing completed.")

# ------------------------------------------------------------------------
# 8) 종료 처리
# ------------------------------------------------------------------------
dist.barrier()
dist.destroy_process_group()
