import sys
from pathlib import Path
import torch
import json
import time
import numpy as np
import argparse
import gc
import subprocess
from transformers import DynamicCache
from tqdm import tqdm
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import WhisperFeatureExtractor

# From trainer: 커스텀 모듈 경로 추가
sys.path.insert(0, str((Path(__file__).parent / "audiolm-trainer").resolve()))
from config import Config
from dataset import SALMONNDataset
from utils import get_dataloader, prepare_sample
from models.salmonn import SALMONN


def load_model(salmonn_preprocessor):
    model = salmonn_preprocessor.llama_model
    tokenizer = salmonn_preprocessor.llama_tokenizer
    return model, tokenizer


def load_preprocessor(cfg):
    salmonn_preprocessor = SALMONN.from_config(cfg.config.model)
    # 모델을 명시적으로 cfg.config.run.device로 이동
    salmonn_preprocessor.to(cfg.config.run.device)
    salmonn_preprocessor.eval()
    return salmonn_preprocessor


class MockDataset(SALMONNDataset):
    def __init__(self, cfg, sr, audio_length, dataset_length):
        self.sr = sr
        self.audio_length = audio_length
        self.dataset_length = dataset_length
        self.prefix = cfg.config.datasets.prefix
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(
            cfg.config.datasets.whisper_path
        )
        self.random_sample = np.random.randn(self.sr * self.audio_length)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        audio = self.random_sample.copy()
        spectrogram = self.wav_processor(
            audio, sampling_rate=self.sr, return_tensors="pt"
        )["input_features"].squeeze()
        return {
            "spectrogram": spectrogram,
            "raw_wav": audio,
            "text": "test",
            "task": "asr",
            "Q": "",
            "id": idx,
        }

    @staticmethod
    def make_mock_dataloader(cfg, sr, audio_length, dataset_length=100):
        dataset = MockDataset(cfg, sr, audio_length, dataset_length)
        return get_dataloader(
            dataset, cfg.config.run, is_train=False, use_distributed=False
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path",
        type=str,
        help="path to configuration file",
        default="salmonn_eval_config.yaml",
    )
    # 기본 디바이스를 "cuda:1" 등 원하는 디바이스로 설정 (예: V100이 할당된 디바이스)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file",
    )
    parser.add_argument("--num_it", type=int, default=100)
    parser.add_argument("--num_warmup", type=int, default=10)
    return parser.parse_args()


def get_gpu_memory_usage():
    result = subprocess.check_output(
        ["nvidia-smi", "-i", "1", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    gpu_memory = int(result.strip().split("\n")[0])
    return gpu_memory


def model_inference(cfg, samples, test_prompt, salmonn):
    """
    개선된 추론 함수 (V100 최적화):
    - torch.inference_mode와 torch.autocast를 사용해 FP16 혼합 정밀도 적용
    - torch.cuda.synchronize()를 통해 GPU 작업 완료를 보장하여 정확한 타이밍 측정
    """
    llm = salmonn.llama_model

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            torch.cuda.synchronize()
            start_time = time.time()

            batch_size = samples["spectrogram"].shape[0]
            spectrogram = samples["spectrogram"]
            raw_wav = samples.get("raw_wav", None)
            audio_padding_mask = samples.get("padding_mask", None)
            speech_embeds, speech_atts = salmonn.encode_speech(
                spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
            )

            prompts = [test_prompt[task] for task in samples["task"]]
            templated_prompts = [
                cfg.config.model.prompt_template.format(prompt) for prompt in prompts
            ]

            speech_embeds, speech_atts = salmonn.prompt_wrap(
                speech_embeds, speech_atts, templated_prompts, multi_prompt=True
            )

            bos = torch.ones(
                [batch_size, 1],
                dtype=torch.int32,
                device=speech_embeds.device,
            ) * salmonn.llama_tokenizer.bos_token_id

            bos_embeds = (
                llm.model.embed_tokens(bos)
                if not salmonn.lora
                else llm.model.model.embed_tokens(bos)
            )
            atts_bos = speech_atts[:, :1]

            speech_embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
            speech_atts = torch.cat([atts_bos, speech_atts], dim=1)

            outputs = llm.model(
                inputs_embeds=speech_embeds,
                attention_mask=speech_atts,
            )
            torch.cuda.synchronize()
            ttft = time.time() - start_time

            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)
            past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)

            torch.cuda.synchronize()
            start_time = time.time()
            _ = llm.model(next_token, past_key_values=past_key_values, use_cache=True)
            torch.cuda.synchronize()
            tpot = time.time() - start_time

    inference_time = ttft + tpot
    return inference_time, ttft, tpot


def main(args):
    # 1. 명시적으로 디바이스 설정 (예: "cuda:1")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    cfg = Config(args)
    # config 내 run.device를 args.device로 재설정하여 일관되게 사용
    cfg.config.run.device = args.device

    print("Force batch size as 1")
    cfg.config.run.batch_size_eval = 1

    # 모델과 전처리기 로드 (모델은 이미 cfg.config.run.device로 이동됨)
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, _ = load_model(salmonn_preprocessor)
    salmonn_preprocessor.llama_model = llama_model

    # torch.compile 호출은 제외합니다.

    # 프롬프트와 데이터셋 로드
    with open("audiolm-trainer/prompts/test_prompt.json", "r") as f:
        test_prompt = json.load(f)
    dataloader = MockDataset.make_mock_dataloader(cfg, sr=16000, audio_length=10)
    sample_batch = next(iter(dataloader))
    sample_batch = prepare_sample(sample_batch, cuda_enabled=torch.cuda.is_available())

    # 메모리 및 추론 시간 측정
    memory_usages = []
    inference_times = []
    ttfts = []
    tpots = []

    for it in range(args.num_it + args.num_warmup):
        torch.cuda.synchronize()
        with torch.inference_mode():
            inference_time, ttft, tpot = model_inference(
                cfg, sample_batch, test_prompt, salmonn_preprocessor
            )
        torch.cuda.synchronize()
        after_memory_allocated = torch.cuda.max_memory_allocated()

        torch.cuda.empty_cache()  # 캐시 비우기
        gc.collect()

        if it >= args.num_warmup:
            memory_usages.append(after_memory_allocated)
            inference_times.append(inference_time)
            ttfts.append(ttft)
            tpots.append(tpot)

    average_memory_usage = np.mean(memory_usages)
    average_inference_time = np.mean(inference_times)
    average_ttft = np.mean(ttfts)
    average_tpot = np.mean(tpots)

    print(f"Average memory used during inference: {average_memory_usage/1024**3:.4f} GB")
    print(f"Average inference time: {average_inference_time:.4f} seconds")
    print(f"Average TTFT: {average_ttft:.4f} seconds")
    print(f"Average TPOT: {average_tpot:.4f} seconds")


if __name__ == "__main__":
    args = parse_args()
    main(args)
