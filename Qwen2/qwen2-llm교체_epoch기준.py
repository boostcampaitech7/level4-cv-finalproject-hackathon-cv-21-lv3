import os
import csv
import json
import datetime
import wandb
import librosa
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import argparse
import pandas as pd
import numpy as np
import soundfile as sf
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from collections import defaultdict
import random

# Constants
PREFIX = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/data"
TRAIN_JSON = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/json/stage1_train_sep.json"
VALID_JSON = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/json/stage1_valid_sep.json"
AAC_JSON = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/json/test_aac.json"
ASR_JSON = "/data/level4-cv-finalproject-hackathon-cv-21-lv3/json/test_asr.json"
AUDIO_CHECKPOINT = "Qwen/Qwen2-Audio-7B"
PROMPT = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"

WANDB_ENTITY = "ppp6131-yonsei-university"
WANDB_PROJECT = "Nota-Qwen2"

class DebugLogger:
    @staticmethod
    def log_dataset_sample(dataset, num_samples=8):
        print("=== Dataset Consistency Check ===")
        for i in range(num_samples):
            sample = dataset[i]  # Dataset에서 개별 샘플 가져오기
            print(f"Sample {i}:")
            print(f"  Audio Path: {sample['id']}")
            print(f"  Audio Shape: {sample['raw_wav'].shape}")
            print(f"  Ground Truth Text: {sample.get('text', 'N/A')}")
            if sample['raw_wav'] is None:
                print(f"  Warning: Missing audio data for {sample['id']}!")
            if sample.get('text') is None or sample.get('text').strip() == "":
                print(f"  Warning: Missing or empty text label for {sample['id']}!")
   
    @staticmethod
    def log_scheduler_learning_rate(scheduler, optimizer):
        lr = scheduler.get_last_lr() if hasattr(scheduler, 'get_last_lr') else [group['lr'] for group in optimizer.param_groups]
        print(f"Current Learning Rate: {lr}")

    @staticmethod
    def log_input_feature_statistics(inputs):
        print("=== Input Feature Statistics ===")
        input_features = inputs['input_features']

        # Input feature의 통계값 계산
        print(f"Input Features - Min: {input_features.min().item()}")
        print(f"Input Features - Max: {input_features.max().item()}")
        print(f"Input Features - Mean: {input_features.mean().item()}")
        print(f"Input Features - Std Dev: {input_features.std().item()}")

        # 첫 번째 샘플의 일부 확인
        print(f"Sample Input Features (First Sequence, First Frame): {input_features[0, 0, :10].tolist()}")


    @staticmethod
    def log_model_inputs(inputs, text_prompts):
        print("=== Input Debugging ===")
        print(f"Input IDs Shape: {inputs['input_ids'].shape}")
        print(f"Attention Mask Shape: {inputs['attention_mask'].shape}")
        print(f"Input Features Shape: {inputs['input_features'].shape}")
        print(f"Feature Attention Mask Shape: {inputs['feature_attention_mask'].shape}")
        print(f"Text Prompts: {text_prompts[:5]} (showing first 5)")

    @staticmethod
    def log_model_outputs(outputs, labels):
        print("=== Model Outputs Debugging ===")
        logits = outputs.logits
        loss = outputs.loss if hasattr(outputs, 'loss') else None

        # 로짓 크기와 일부 값 확인
        print(f"Logits Shape: {logits.shape}")
        print(f"Logits (First Sequence, First Token): {logits[0, 0, :5].tolist()}")

        # Softmax를 사용하여 log-prob 계산
        probs = torch.softmax(logits, dim=-1)
        print(f"Probabilities (First Sequence, First Token): {probs[0, 0, :5].tolist()}")

        # 라벨 및 손실 정보
        print(f"Labels Shape: {labels.shape}")
        print(f"Labels (First Sequence): {labels[0][:5].tolist()}")
        if loss is not None:
            print(f"Loss: {loss.item():.4f}")

    @staticmethod
    def log_random_sample(inputs, outputs, labels):
        print("=== Random Sample Debugging ===")
        batch_size = inputs['input_ids'].shape[0]
        random_idx = random.randint(0, batch_size - 1)

        print(f"Random Sample Index: {random_idx}")
        print(f"Input IDs (Sample): {inputs['input_ids'][random_idx][:10].tolist()}")
        print(f"Attention Mask (Sample): {inputs['attention_mask'][random_idx][:10].tolist()}")
        print(f"Predicted Tokens (Sample): {outputs.logits[random_idx, :, :5]}")
        print(f"Labels (Sample): {labels[random_idx][:10].tolist()}")

    @staticmethod
    def log_learning_rate(optimizer):
        print("=== Learning Rate Debugging ===")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

    @staticmethod
    def log_gradient_norms(model):
        print("=== Gradient Norms Debugging ===")
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                print(f"Param: {name}, Gradient Norm: {grad_norm:.4f}")

    @staticmethod
    def log_epoch_summary(epoch, avg_train_loss, avg_val_loss=None, learning_rate=None):
        print("=== Epoch Summary ===")
        print(f"Epoch: {epoch}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        if avg_val_loss is not None:
            print(f"Validation Loss: {avg_val_loss:.4f}")
        if learning_rate is not None:
            print(f"Learning Rate: {learning_rate:.6f}")

    @staticmethod
    def log_padding_distribution(inputs):
        print("=== Padding Distribution Debugging ===")
        attention_masks = inputs['attention_mask']
        input_ids = inputs['input_ids']

        # 패딩이 적용된 부분의 개수
        padding_counts = torch.sum(attention_masks == 0, dim=1)
        print(f"Padding Counts per Sequence: {padding_counts.tolist()}")

        # 각 시퀀스의 실제 데이터 길이 확인
        sequence_lengths = torch.sum(attention_masks, dim=1)
        print(f"Actual Sequence Lengths: {sequence_lengths.tolist()}")

        # 첫 번째 시퀀스의 ID 값 확인
        print(f"Sample Input IDs (First Sequence): {input_ids[0, :10].tolist()}")
        print(f"Attention Mask (First Sequence): {attention_masks[0, :10].tolist()}")

    @staticmethod
    def log_audio_statistics(audio_inputs):
        print("=== Audio Data Statistics ===")
        print(f"Min Value: {audio_inputs.min().item()}")
        print(f"Max Value: {audio_inputs.max().item()}")
        print(f"Mean: {audio_inputs.mean().item()}")
        print(f"Std Dev: {audio_inputs.std().item()}")


def load_lora_and_adapter_weights(model, checkpoint_path):
    """
    Load only the LoRA and Adapter weights from a checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.module.language_model.load_state_dict(checkpoint["lora"], strict=False)
    model.module.multi_modal_projector.load_state_dict(checkpoint["adapter"], strict=False)
    print("Loaded LoRA and Adapter weights.")

def custom_collate_fn(batch):
    # 오디오 텐서를 패딩
    audio_tensors = [item["raw_wav"] for item in batch]
    padded_audio = pad_sequence(audio_tensors, batch_first=True)

    # 나머지 데이터는 그대로 반환
    batch_data = {
        "raw_wav": padded_audio,
        "task": [item["task"] for item in batch],
        "Q": [item["Q"] for item in batch],
        "id": [item["id"] for item in batch],
    }

    # 텍스트 데이터 처리 (있는 경우에만)
    if "text" in batch[0]:
        batch_data["text"] = [item["text"] for item in batch]

    return batch_data

class Adapter(nn.Module):
    def __init__(self, adapter_type="bottleneck", input_dim=4096, hidden_dim=512, output_dim=1536, dropout=0.1):
        super(Adapter, self).__init__()
        self.adapter_type = adapter_type

        if adapter_type == "bottleneck":
            self.adapter = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
            self.skip_connection = nn.Linear(input_dim, output_dim)
        elif adapter_type == "mlp":
            self.adapter = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        if self.adapter_type == "bottleneck":
            x = self.adapter(x) + self.skip_connection(x)
        else:
            x = self.adapter(x)
        x = self.layer_norm(x)
        return x

def save_checkpoint(model, epoch, save_dir, best_loss):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}_loss_{best_loss:.4f}.pth")
    torch.save({
        "lora": model.module.language_model.state_dict(),
        "adapter": model.module.multi_modal_projector.state_dict(),
    }, checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")

def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def replace_multi_modal_projector_with_adapter(audio_model, llm_model, adapter_type="bottleneck"):
    input_dim = audio_model.multi_modal_projector.linear.weight.shape[1]
    output_dim = llm_model.config.hidden_size

    adapter = Adapter(adapter_type=adapter_type, input_dim=input_dim, hidden_dim=2048, output_dim=output_dim)
    adapter.apply(initialize_weights)

    audio_model.multi_modal_projector = adapter
    audio_model.language_model = llm_model

    return audio_model

# Exclude `language_model` weights from Qwen2-Audio
def exclude_llm_weights(state_dict, llm_prefix="language_model"):
    """
    Exclude the `language_model` weights while loading Qwen2-Audio checkpoint.
    """
    return {k: v for k, v in state_dict.items() if not k.startswith(llm_prefix)}


# Dataset class
class Qwen2AudioDataset(Dataset):
    def __init__(self, prefix, ann_path, task=None):
        self.prefix = prefix
        self.annotation = json.load(open(ann_path, "r"))["annotation"][:8000]
        self.task = task

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        if ann["path"].startswith("/"):  # 절대 경로인 경우
            ann["path"] = ann["path"][1:]  # 맨 앞의 "/" 제거 
        audio_path = os.path.join(self.prefix, ann["path"])

        try:
            audio, sr = sf.read(audio_path)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except Exception as e:
            print(f"Failed to load {audio_path}. Error: {e}. Using silent audio as fallback.")
            audio = torch.zeros(16000)  # 1 second of silent audio at 16 kHz
            sr = 16000

        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        if len(audio) < sr:
            audio = np.pad(audio, (0, sr - len(audio)), mode="constant")
        audio = audio[: sr * 30]
        audio = torch.tensor(audio, dtype=torch.float32)

        entity = {
            "raw_wav": audio,
            "task": ann.get("task", "asr"),
            "Q": ann.get("Q", ""),
            "id": ann["path"],
        }

        if self.task is not None and "text" in ann:
            entity["text"] = ann["text"]

        return entity

def set_seed(seed: int):
    """
    Set the seed for reproducibility across various libraries.
    
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy's random module
    torch.manual_seed(seed)  # PyTorch CPU random numbers
    torch.cuda.manual_seed(seed)  # PyTorch GPU random numbers
    torch.cuda.manual_seed_all(seed)  # All GPUs (if using multi-GPU)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic algorithms
    torch.backends.cudnn.benchmark = False  # Turn off optimizations for reproducibility

    print(f"Seed set to: {seed}")

# Learning rate scheduler
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch

        # Initialize _last_lr attribute
        self._last_lr = [group['lr'] for group in optimizer.param_groups]

        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) *
                    (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, step_increment=1):
        self.T_cur += step_increment
        if self.T_cur >= self.T_i:
            self.cycle += 1
            self.T_cur = self.T_cur - self.T_i
            self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch += step_increment

        # Update _last_lr with the new learning rates
        self._last_lr = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group['lr'] = lr



## Training loop ##
def train_loop(
    model, train_dataset, valid_loader, optimizer, scheduler, processor, device, epochs,
    save_dir="checkpoints", patience=5, wandb_run=None, max_iterations_per_epoch=60000, batch_size=4):
    
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    patience_counter = 0
    scaler = GradScaler()  # Mixed Precision을 위한 Scaler 추가
    
    patience_countter = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, collate_fn=custom_collate_fn
        )
        
        DebugLogger.log_dataset_sample(train_dataset)

        # Training phase
        progress_bar = tqdm((train_loader), total=len(train_loader), desc=f"[{epoch + 1}/{epochs}] Training")
        for batch in progress_bar:
            
            audio_inputs = batch["raw_wav"].to(device)
            text_prompts = [PROMPT] * len(audio_inputs)

            DebugLogger.log_audio_statistics(audio_inputs)

            optimizer.zero_grad()
            # Processor를 사용하여 모델 입력 생성
            with autocast(device_type='cuda'):
                inputs = processor(
                    text=text_prompts,
                    audios=[audio.cpu().numpy() for audio in audio_inputs],
                    return_tensors="pt",
                    sampling_rate=16000,
                ).to(device)
            
            DebugLogger.log_model_inputs(inputs, text_prompts)
            DebugLogger.log_input_feature_statistics(inputs)
            DebugLogger.log_padding_distribution(inputs)
            
            with autocast(device_type='cuda'):
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

            DebugLogger.log_model_outputs(outputs, inputs["input_ids"])
            DebugLogger.log_random_sample(inputs, outputs, inputs["input_ids"])

            scaler.scale(loss).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            DebugLogger.log_gradient_norms(model)

            # Optimizer Step with Scaler
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

            DebugLogger.log_scheduler_learning_rate(scheduler, optimizer)

            # Update tqdm progress bar
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}",
            )
                
        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        DebugLogger.log_learning_rate(optimizer)


        # Log training loss to WandB (only once per epoch)
        if dist.get_rank() == 0 and wandb_run is not None:
            wandb_run.log(
                {
                    "Epoch": epoch + 1,
                    "Avg Training Loss": avg_train_loss,
                    "Learning Rate": optimizer.param_groups[0]['lr'],
                }
            )

        # Gather training losses from all ranks
        train_loss_tensor = torch.tensor(avg_train_loss, device=device)
        all_train_losses = [torch.zeros(1, device=device) for _ in range(dist.get_world_size())]
        dist.all_gather(all_train_losses, train_loss_tensor)

        ## Validation phase ##
        if (epoch + 1) % 1 == 0:
            if valid_loader:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(valid_loader, desc=f"[{epoch + 1}/{epochs}] Validation"):
                        audio_inputs = batch["raw_wav"].to(device)
                        text_prompts = [PROMPT] * len(audio_inputs)

                        inputs = processor(
                            text=text_prompts,
                            audios=[audio.cpu().numpy() for audio in audio_inputs],
                            return_tensors="pt",
                            sampling_rate=16000,
                        ).to(device)

                        outputs = model(**inputs, labels=inputs["input_ids"])
                        val_loss += outputs.loss.item()

                    
                avg_val_loss = val_loss / len(valid_loader)
                print(f"Rank {dist.get_rank()}, Device {device}: Validation Loss: {avg_val_loss:.4f}")

                # Gather validation losses from all ranks
                val_loss_tensor = torch.tensor(avg_val_loss, device=device)
                all_val_losses = [torch.zeros(1, device=device) for _ in range(dist.get_world_size())]
                dist.all_gather(all_val_losses, val_loss_tensor)

                # Rank 0: Log to WandB with separate fields for each rank
                if dist.get_rank() == 0:
                    log_data = {"Epoch": epoch + 1, "Learning Rate": optimizer.param_groups[0]['lr']}
                    for rank_id, train_loss in enumerate(all_train_losses):
                        log_data[f"Training Loss (Rank {rank_id})"] = train_loss.item()
                    for rank_id, val_loss in enumerate(all_val_losses):
                        log_data[f"Validation Loss (Rank {rank_id})"] = val_loss.item()

                    wandb_run.log(log_data)

                # Determine the rank with the smallest validation loss
                all_val_losses = [t.item() for t in all_val_losses]
                min_loss = min(all_val_losses)
                min_loss_rank = all_val_losses.index(min_loss)

                if dist.get_rank() == min_loss_rank:
                    if avg_val_loss < best_loss- 1e-3:  # Use threshold for significant improvement
                        best_loss = avg_val_loss
                        patience_counter = 0

                        if dist.get_rank() == 0:
                            checkpoint_path = os.path.join(save_dir, f"lora_adapter_epoch_{epoch + 1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
                            save_checkpoint(model, epoch, checkpoint_path, best_loss)

                    else:
                        patience_countter += 1
                        print(f"No significant improvement. Early stopping counter: {patience_counter}/{patience}")

                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break

        DebugLogger.log_epoch_summary(epoch + 1, avg_train_loss, avg_val_loss, optimizer.param_groups[0]['lr'])


# Inference function
def inference_loop(model, processor, json_path, output_csv, device="cuda"):

    try:
        with open(json_path, "r") as f:
            data = json.load(f)["annotation"]

        results = []
        for item in tqdm(data, desc="Inference"):
            testset_id = item["testset_id"]
            audio_path = os.path.join(PREFIX, item["path"])

            try:
                audio, sr = sf.read(audio_path)
                if audio.ndim == 2:
                    audio = audio.mean(axis=0)
                # audio = audio[: sr * 30]
            except Exception as e:
                print(f"Failed to load {audio_path}. Using silent audio. Error: {e}.")
                audio = torch.zeros(16000)

            inputs = processor(text=PROMPT, audios=audio, return_tensors="pt", sampling_rate=16000).to(device)

            with torch.no_grad():
                outputs = model.module.generate(**inputs)  # Access module directly
            text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            results.append({"testset_id": testset_id, "text": text})

        # 오름차순 정렬 후 저장
        results.sort(key=lambda x: x["testset_id"])
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["testset_id", "text"])
            writer.writeheader()
            writer.writerows(results)

    finally:
        # Destroy process group
        if dist.is_initialized():
            dist.destroy_process_group()


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["train", "inference"], required=True)
    parser.add_argument("--llm_checkpoint", default="Qwen/Qwen2-1.5B-Instruct", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, help="Path to the LoRA and Adapter checkpoint for inference")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=40)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--wandb_name", type=str, default="Qwen2-1.5B", help="Name for the wandb logging")
    parser.add_argument("--max_iterations_per_epoch", type=int, default=1500, help="Max iterations per epoch")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "True"

    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    
    seed =7
    set_seed(seed)
    
    # Debug: Log environment variables
    print("Environment Variables:")
    for key, value in os.environ.items():
        if "RANK" in key or "WORLD" in key or "LOCAL" in key:
            print(f"{key}: {value}")

    # Get local_rank
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    print(f"Local rank detected: {local_rank}")

    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        print(f"Invalid local_rank {local_rank}. Available GPUs: {torch.cuda.device_count()}")
        raise ValueError(f"Invalid local_rank {local_rank}. Check your distributed setup.")
    
    # Set CUDA device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"Using device: {device}")

    # Load model
    audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        AUDIO_CHECKPOINT,
        state_dict=exclude_llm_weights(
            torch.load(f"qwen2_audio_7b_state_dict.bin", map_location="cpu", weights_only=True)
        ),
        trust_remote_code=True,
    )
    for param in audio_model.audio_tower.parameters():
        param.requires_grad = False

    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_checkpoint, trust_remote_code=True)
    for param in llm_model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, task_type=TaskType.CAUSAL_LM)
    llm_model = get_peft_model(llm_model, lora_config)
    audio_model = replace_multi_modal_projector_with_adapter(audio_model, llm_model, adapter_type="bottleneck")
    audio_model = audio_model.to(device)    
    audio_model = DDP(audio_model, device_ids=[local_rank], output_device=local_rank)

    processor = AutoProcessor.from_pretrained(AUDIO_CHECKPOINT, trust_remote_code=True)

    if args.task == "train":
        
        # Initialize wandb
        if dist.get_rank() == 0:

            wandb_run = wandb.init(
                project="Nota-Qwen2",
                entity="ppp6131-yonsei-university",
                name=f"{args.wandb_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args),
            )
        else:
            wandb_run = None  # Ensure other ranks don't use WandB
        train_dataset = Qwen2AudioDataset(PREFIX, TRAIN_JSON, task="train")
        valid_dataset = Qwen2AudioDataset(PREFIX, VALID_JSON, task="valid")
    
        # train_sampler = DistributedSampler(train_dataset, shuffle=True)
        # valid_sampler = DistributedSampler(valid_dataset, shuffle=False)

        # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=16, pin_memory=True, prefetch_factor=4, collate_fn=custom_collate_fn)
        valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, num_workers=16, pin_memory=True, prefetch_factor=4, collate_fn=custom_collate_fn)

        optimizer = torch.optim.AdamW(
            list(audio_model.module.language_model.parameters()) +
            list(audio_model.module.multi_modal_projector.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
        )   

        total_steps = args.max_iterations_per_epoch * args.epochs
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0=6,                  # 6 epoch마다 1번의 Cosine Annealing 주기
            T_mult=1,               # 이후 주기도 동일 (T_mult=1로 고정)
            eta_max=1e-4,           # 최대 학습률
            T_up=2,                 # Warmup 기간: 2 epoch
            gamma=0.8               # 이후 주기 감쇠 계수
        )
        train_loop(audio_model, train_dataset, valid_loader, optimizer, scheduler, processor, device, args.epochs, wandb_run=wandb_run, max_iterations_per_epoch=args.max_iterations_per_epoch, batch_size=args.batch_size)
        if dist.get_rank() == 0:
            wandb_run.finish()
        
    elif args.task == "inference":
        if not args.checkpoint_path:
            raise ValueError("Please provide --checkpoint_path for inference.")

        if dist.get_rank() == 0:
            load_lora_and_adapter_weights(audio_model, args.checkpoint_path)
            inference_loop(audio_model, processor, AAC_JSON, "results/qwen_aac.csv", device)
            inference_loop(audio_model, processor, ASR_JSON, "results/qwen_asr.csv", device)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()