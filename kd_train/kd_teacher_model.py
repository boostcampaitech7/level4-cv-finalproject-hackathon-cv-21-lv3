import torch
import contextlib
import logging
import yaml
import os

import torch.nn.functional as F
import torch.nn as nn
from model import SALMONN
from models.modeling_whisper import WhisperModel
from models.beats.BEATs import BEATs, BEATsConfig
from models.Qformer import BertLMHeadModel, BertConfig
from transformers import StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, GPTQConfig, QuantoConfig


class TeacherModel(torch.nn.Module):
    def __init__(self, 
                 ckpt_path="/data/audiolm-evaluator/audiolm-trainer/checkpoint_salmmon/salmonn_7b_v0.pth",
                 whisper_path="openai/whisper-large-v2",
                 beats_path="/data/audiolm-evaluator/audiolm-trainer/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
                 vicuna_path="lmsys/vicuna-7b-v1.5",
                 lora_alpha=32,
                 low_resource=True,
                 device=None):
        super().__init__()
        
        if device is None:
            self.device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        else:
            self.device = torch.device(device)
        
        # 모델 초기화
        self.model = SALMONN(
            ckpt=ckpt_path,
            whisper_path=whisper_path,
            beats_path=beats_path,
            vicuna_path=vicuna_path,
            lora_alpha=lora_alpha,
            low_resource=low_resource,
            device=self.device,
            quantization="4bit"
        )
            
    def forward(self, samples):
        """Teacher Model은 단순히 SALMONN의 forward를 호출하여 logits을 반환"""
        with torch.no_grad(): 
            return self.model(samples)