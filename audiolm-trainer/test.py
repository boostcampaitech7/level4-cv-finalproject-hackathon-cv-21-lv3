import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 모델 ID (Hugging Face에 등록된 모델 이름)
model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 모델 로드 (모델은 사전에 GPTQ와 SpinQuant 방법으로 INT4 양자화되어 있음)
# device_map="auto" 옵션으로 사용 가능한 GPU를 자동 할당합니다.
bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,      # 계산 시 사용 dtype (예: fp16)
                    bnb_4bit_quant_type="nf4",                   # NF4 방식 사용 (또는 필요에 따라 "fp4" 선택)
                    bnb_4bit_use_double_quant=True,              # double quantization 적용 (메모리 절감 효과)
                    bnb_4bit_group_size=32,                # 그룹 사이즈 32를 사용하여 그룹 와이즈 양자화
                    # 분류 계층(lm_head)와 임베딩 계층(embed_tokens)은 4-bit 양자화에서 제외하여
                    # 기본 8-bit per-channel 방식이 적용되도록 함
                    llm_int8_skip_modules=["lm_head", "embed_tokens"]
                )
                
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16  # 필요에 따라 torch.bfloat16로 변경 가능
)

# 간단한 프롬프트 예제
prompt = "안녕하세요. 오늘 날씨는 어떤가요?"

# 프롬프트 토크나이즈 및 모델 장치로 이동
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# GPU가 사용 가능하면, peak memory 통계를 초기화합니다.
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# 텍스트 생성 (최대 100 토큰)
outputs = model.generate(**inputs, max_new_tokens=1024)

# GPU 사용 시, peak memory 사용량을 MB 단위로 변환하여 출력합니다.
if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated(model.device) / (1024 ** 2)
    print(f"모델 추론 중 peak 메모리 사용량: {peak_memory:.2f} MB")


# 생성된 토큰을 디코딩하여 출력
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
