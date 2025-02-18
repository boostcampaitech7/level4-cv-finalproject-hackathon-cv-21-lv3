import os
import json
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from dataset import Qwen2AudioDataset

# Paths
PROMPT_JSON_PATH = "/data/audiolm-evaluator/audiolm-trainer/prompts/test_prompt.json"
RESULT_AAC_CSV = "results/qwen_aac.csv"
RESULT_ASR_CSV = "results/qwen_asr.csv"
AAC_JSON = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/json/test_aac.json"
ASR_JSON = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/json/test_asr.json"
CHECKPOINT = "Qwen/Qwen2-Audio-7B"
PREFIX = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/data"


def load_prompts(prompt_path):
    """
    Load task-specific prompts from the JSON file.
    """
    with open(prompt_path, "r") as f:
        prompts = json.load(f)
    
    # Replace the keyword in each prompt
    transformed_prompts = {
        "asr": "<|audio_bos|><|AUDIO|><|audio_eos|> Recognize the speech and give me the transcription.",
        "audiocaption": "<|audio_bos|><|AUDIO|><|audio_eos|> Please describe the audio in English."
    }    
    return transformed_prompts



def evaluate_and_save_results(model, processor, dataloader, output_csv, task_name, prompt, device):
    print(f"Running evaluation for {task_name}...")

    results = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {task_name}"):
            testset_ids = batch["testset_id"]
            raw_wavs = batch["raw_wav"]

            # Ensure raw_wavs are numpy arrays if needed
            if isinstance(raw_wavs[0], torch.Tensor):
                raw_wavs = [wav.numpy() for wav in raw_wavs]

            for i, (testset_id, audio) in enumerate(zip(testset_ids, raw_wavs)):
                print(f"Processing audio {i} with testset_id: {testset_id}, Length = {len(audio)}")
                
                # Prepare single audio input
                inputs = processor(
                    text=[prompt],
                    audios=[audio],
                    return_tensors="pt",
                    sampling_rate=16000,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                print(f'prompt is {prompt}')
                # Debugging inputs
                # print(f"Input IDs shape: {inputs['input_ids'].shape}")
                # print(f"Input Features shape: {inputs['input_features'].shape}")
                # print(f"Feature Attention Mask shape: {inputs['feature_attention_mask'].shape}")

                # Generate prediction
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    input_features=inputs["input_features"],
                    feature_attention_mask=inputs["feature_attention_mask"],
                    max_length=256,
                )

                # Decode output
                decoded_output = processor.batch_decode(output, skip_special_tokens=True)
                transcription = decoded_output[0] if decoded_output else "[No transcription]"
                if transcription.strip() == prompt.strip():
                    transcription = "[Empty or Invalid Audio]"

                print(f"Decoded output for {testset_id}: {transcription}")
                
                # Append result
                results.append({"testset_id": testset_id, "text": transcription})

    # Sort results by testset_id
    results = sorted(results, key=lambda x: x["testset_id"])
    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for computation (e.g., 'cuda:0', 'cpu')")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model and processor
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        CHECKPOINT, device_map=device, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(CHECKPOINT)

    # Load prompts
    prompts = load_prompts(PROMPT_JSON_PATH)

    # Evaluate ASR
    asr_dataset = Qwen2AudioDataset(PREFIX, ASR_JSON, CHECKPOINT, task="asr")
    asr_dataloader = DataLoader(
        asr_dataset, 
        batch_size=4,  
        collate_fn=asr_dataset.collater,  
        shuffle=True  
    )    
    evaluate_and_save_results(model, processor, asr_dataloader, RESULT_ASR_CSV, "asr", prompts["asr"], device)

    # Evaluate AAC
    aac_dataset = Qwen2AudioDataset(PREFIX, AAC_JSON, CHECKPOINT, task="aac")
    aac_dataloader = DataLoader(
        aac_dataset, 
        batch_size=4,  
        collate_fn=aac_dataset.collater,  
        shuffle=True  
    ) 
    evaluate_and_save_results(model, processor, aac_dataloader, RESULT_AAC_CSV, "aac", prompts["audiocaption"], device)