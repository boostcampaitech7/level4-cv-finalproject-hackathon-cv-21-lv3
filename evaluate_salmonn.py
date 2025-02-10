import argparse
import json
import random
import sys
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add custom module path
sys.path.insert(0, str((Path(__file__).parent / "audiolm-trainer").resolve()))

# Custom modules
from salmonn_utils import SALMONNTestDataset, load_preprocessor, load_model
from config import Config
from utils import get_dataloader, prepare_sample
from metrics import compute_wer, compute_spider


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path",
        type=str,
        help='path to configuration file',
        default='salmonn_eval_config.yaml'
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    parser.add_argument("--task", type=str, default=None, help="Task to evaluate", choices=['asr', 'aac'])
    parser.add_argument("--skip_scoring", action='store_true', help='if True, skip scoring after inference')
    return parser.parse_args()


def get_dataset(dataset_cfg, run_cfg, task):
    testset = SALMONNTestDataset(
        dataset_cfg.prefix, dataset_cfg.test_ann_path, dataset_cfg.whisper_path, task
    )
    # DataLoader에 pin_memory=True를 사용하여 GPU로 데이터를 더 빠르게 전송
    test_loader = get_dataloader(testset, run_cfg, is_train=False, use_distributed=False, pin_memory=True)
    return test_loader


def main(args):
    cfg = Config(args)

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # 모델과 전처리기 로드
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, tokenizer = load_model(salmonn_preprocessor)

    # 평가 모드로 전환하고 device로 이동
    llama_model.eval()
    llama_model.to(device)
    if hasattr(salmonn_preprocessor, "to"):
        salmonn_preprocessor.to(device)
    salmonn_preprocessor.llama_model = llama_model

    # 데이터 로드
    dataloader = get_dataset(cfg.config.datasets, cfg.config.run, args.task)

    with open("audiolm-trainer/prompts/test_prompt.json", "r") as f:
        test_prompt = json.load(f)

    testset_ids, hyps, refs = [], [], []

    # 추론 시 불필요한 gradient 계산을 제거하고, FP16 혼합 정밀도 사용
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for samples in tqdm(dataloader):
            testset_ids.extend(samples["testset_id"])

            # 샘플 전처리 (내부에서 device 이동 등을 처리한다고 가정)
            samples = prepare_sample(samples, cuda_enabled=torch.cuda.is_available())
            batch_size = samples["spectrogram"].shape[0]
            spectrogram = samples["spectrogram"]
            raw_wav = samples.get("raw_wav", None)
            audio_padding_mask = samples.get("padding_mask", None)

            # 음성 인코딩
            speech_embeds, speech_atts = salmonn_preprocessor.encode_speech(
                spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
            )

            # 프롬프트 적용
            prompts = [test_prompt[task] for task in samples['task']]
            templated_prompts = [cfg.config.model.prompt_template.format(prompt) for prompt in prompts]
            speech_embeds, speech_atts = salmonn_preprocessor.prompt_wrap(
                speech_embeds, speech_atts, templated_prompts, multi_prompt=True
            )

            bos = torch.ones(
                [batch_size, 1],
                dtype=torch.int32,
                device=speech_embeds.device,
            ) * tokenizer.bos_token_id
            bos_embeds = llama_model.model.model.embed_tokens(bos)
            atts_bos = speech_atts[:, :1]

            embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
            attns = torch.cat([atts_bos, speech_atts], dim=1)

            generate_cfg = cfg.config.generate

            outputs = llama_model.model.generate(
                inputs_embeds=embeds,
                pad_token_id=llama_model.config.eos_token_id,
                max_new_tokens=generate_cfg.get("max_new_tokens", 200),
                num_beams=generate_cfg.get("num_beams", 4),
                do_sample=generate_cfg.get("do_sample", False),
                min_length=generate_cfg.get("min_length", 1),
                temperature=generate_cfg.get("temperature", 1.0),
                top_p=generate_cfg.get("top_p", 0.9),
                repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
                length_penalty=generate_cfg.get("length_penalty", 1.0),
                attention_mask=attns,
            )

            results = tokenizer.batch_decode(outputs)
            hyp = [result.split(generate_cfg.end_sym)[0].lower() for result in results]
            hyps.extend(hyp)

            if not args.skip_scoring:
                refs.extend(samples["text"])

    if not args.skip_scoring:
        if args.task == 'asr':
            compute_wer(hyps, refs)
        elif args.task == 'aac':
            compute_spider(hyps, refs)

    result_df = pd.DataFrame({"testset_id": testset_ids, "text": hyps})
    result_df.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    args = parse_args()
    random.seed(42)
    main(args)
