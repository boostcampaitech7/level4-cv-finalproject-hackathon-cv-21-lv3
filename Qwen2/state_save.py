import torch
from transformers import Qwen2AudioForConditionalGeneration

# 모델 경로 및 저장 파일 이름
MODEL_CHECKPOINT = "Qwen/Qwen2-Audio-7B"
SAVE_PATH = "qwen2_audio_7b_state_dict.bin"

# 모델 로드
print(f"Loading model from {MODEL_CHECKPOINT}...")
model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT, trust_remote_code=True)

# 모델의 state_dict 저장
print(f"Saving state_dict to {SAVE_PATH}...")
torch.save(model.state_dict(), SAVE_PATH)

print(f"State_dict saved successfully to {SAVE_PATH}.")
