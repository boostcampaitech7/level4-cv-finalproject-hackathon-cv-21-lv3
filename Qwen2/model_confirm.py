import pandas as pd
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2AudioForConditionalGeneration,
)

# Constants
CHECKPOINT = "Qwen/Qwen2-Audio-7B"
NEW_LLM_CHECKPOINT = "Qwen/Qwen2-1.5B-Instruct"  # 변경할 LLM

# Function to calculate model parameters
def get_model_structure(model, model_name):
    layers = []
    total_params = 0

    for name, param in model.named_parameters():
        layer_params = param.numel()
        layers.append({"Model": model_name, "Layer Name": name, "Number of Parameters": layer_params})
        total_params += layer_params

    return layers, total_params

# Load the original model
print(f"Loading model from {CHECKPOINT}...")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    CHECKPOINT,
    trust_remote_code=True,
)

# Calculate parameters for the original model
print(f"Calculating parameters for {CHECKPOINT}...")
original_layers, original_total_params = get_model_structure(model, CHECKPOINT)

# Replace the LLM in the model
print(f"Replacing language_model with {NEW_LLM_CHECKPOINT}...")
llm_config = AutoConfig.from_pretrained(NEW_LLM_CHECKPOINT, trust_remote_code=True)
new_llm = AutoModelForCausalLM.from_pretrained(
    NEW_LLM_CHECKPOINT,
    config=llm_config,
    trust_remote_code=True,
)
model.language_model = new_llm

# Calculate parameters for the modified model
print(f"Calculating parameters for {NEW_LLM_CHECKPOINT}...")
new_llm_layers, new_llm_total_params = get_model_structure(new_llm, NEW_LLM_CHECKPOINT)

# Combine the results into DataFrames
original_df = pd.DataFrame(original_layers)
new_df = pd.DataFrame(new_llm_layers)

# Summary DataFrame
summary_df = pd.DataFrame({
    "Model": [CHECKPOINT, NEW_LLM_CHECKPOINT],
    "Total Parameters": [original_total_params, new_llm_total_params]
})

# Display the results
print(f"\n{'='*30} TOTAL PARAMETERS {'='*30}")
print(summary_df)

print(f"\n{'='*30} {CHECKPOINT} STRUCTURE {'='*30}")
print(original_df)

print(f"\n{'='*30} {NEW_LLM_CHECKPOINT} STRUCTURE {'='*30}")
print(new_df)

# Save detailed layer information to CSV
original_csv = f"model_structure/{CHECKPOINT.replace('/', '_')}_structure.csv"
new_csv = f"model_structure/{NEW_LLM_CHECKPOINT.replace('/', '_')}_structure.csv"
summary_csv = "model_structure/model_summary.csv"

original_df.to_csv(original_csv, index=False)
new_df.to_csv(new_csv, index=False)
summary_df.to_csv(summary_csv, index=False)

print(f"\nDetailed layer information saved to:\n- {original_csv}\n- {new_csv}")
print(f"Summary saved to: {summary_csv}")


print('Audio-7B-multimodal_projector')
for name, module in model.multi_modal_projector.named_children():
    print(f"{name}: {module}")