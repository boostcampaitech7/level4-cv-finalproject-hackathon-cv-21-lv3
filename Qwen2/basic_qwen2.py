import json
import csv
import os
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from scipy.signal import resample_poly
import soundfile as sf

# transformers / bitsandbytes
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig

# 경로 설정
RESULT_AAC_CSV = "results/qwen_aac.csv"
RESULT_ASR_CSV = "results/qwen_asr.csv"
AAC_JSON = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/json/test_aac.json"
ASR_JSON = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/json/test_asr.json"
PREFIX = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/data"

# 로컬에 git clone 받아온 Qwen2-Audio-7B 모델 경로
local_model_path = "./Qwen2-Audio-7B"

##################################################
# 1) 4-bit 양자화 Config 세팅
##################################################
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

##################################################
# 2) 모델 로드 (device_map="auto", offload_folder 지정)
##################################################
print("Loading Qwen2AudioForConditionalGeneration in 4-bit with device_map='auto'...")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    local_model_path,
    trust_remote_code=True,
    quantization_config=bnb_config_4bit,  # 4-bit 양자화 적용
    device_map="auto",                    # 자동으로 GPU/CPU 분산
    max_memory={
        0: "32GiB",    # GPU0에 할당 가능한 메모리
        1: "32GiB",    # GPU1에 할당 가능한 메모리
        "cpu": "120GiB"  # CPU에 할당 가능한 메모리
    },
    offload_folder="./offload_temp"       # 부족 시 디스크로 오프로딩
)

##################################################
# 3) Processor도 로컬 폴더에서 로드
##################################################
print("Loading processor from local path...")
processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)

print("Model and processor loaded successfully.")

##################################################
# (Pruning 로직은 제거했습니다.)
#  - 4-bit 상태에서 바로 Pruning하면 메모리 폭증/오류 가능성.
#  - 필요하면 FP16/FP32 모델에서 Pruning -> 이후 4bit 양자화 과정을 고려하세요.
##################################################

##################################################
# 4) 오디오 처리 함수
##################################################
def process_audio_file(file_path, prompt, max_duration=10, gpu_id="cuda:0"):
    # 사운드 파일 로드
    audio, sr = sf.read(file_path)
    # SR이 16kHz가 아니라면 16kHz로 리샘플링
    if sr != 16000:
        audio = resample_poly(audio, up=16000, down=sr)
        sr = 16000

    # max_duration (초)만큼 오디오 잘라냄
    audio = audio[: sr * max_duration]

    # Processor로 입력 변환
    inputs = processor(text=[prompt], audios=audio, return_tensors="pt").to(gpu_id)

    with autocast(device_type="cuda", dtype=torch.float16):
        with torch.no_grad():
            # generate 호출
            generated_ids = model.generate(**inputs, max_length=128, use_cache=False)
    # 프롬프트 부분을 제외하고 decoding
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response

##################################################
# 5) JSON 로드 -> 결과 CSV 기록 함수
##################################################
def process_json(json_path, result_csv, batch_size=1, gpu_id="cuda:0"):
    print(f"Processing JSON file: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    with open(result_csv, 'w', newline='') as csvfile:
        fieldnames = ['testset_id', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        annotations = data['annotation']
        for i in tqdm(range(0, len(annotations), batch_size), desc="Processing audio files"):
            batch = annotations[i:i + batch_size]
            for item in batch:
                testset_id = item['testset_id']
                audio_path = os.path.join(PREFIX, item['path'])
                prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
                try:
                    transcription = process_audio_file(audio_path, prompt, gpu_id=gpu_id)
                    writer.writerow({'testset_id': testset_id, 'text': transcription})
                except RuntimeError as e:
                    print(f"Error processing {audio_path}: {e}")
                finally:
                    # 매 파일 처리 후 캐시 비우기 (메모리 회수)
                    torch.cuda.empty_cache()

##################################################
# 6) JSON 파일 처리 (AAC, ASR)
##################################################
print("Processing test_aac.json...")
process_json(AAC_JSON, RESULT_AAC_CSV, batch_size=1, gpu_id="cuda:0")

print("Processing test_asr.json...")
process_json(ASR_JSON, RESULT_ASR_CSV, batch_size=1, gpu_id="cuda:0")

print("Processing completed.")
