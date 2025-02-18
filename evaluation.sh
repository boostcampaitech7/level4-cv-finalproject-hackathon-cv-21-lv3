#!/bin/bash

# Step 0
python evaluate_salmonn.py --test_ann_path "/data/level4-cv-finalproject-hackathon-cv-21-lv3/json/test_asr.json" --result_name "submission_mlxcommunity/Llama-3.2-3B-Instruct-4bit_asr" --sample_fp16 True --device cuda:1 # "submission_fp16_asr"
# python evaluate_salmonn.py --test_ann_path "/data/level4-cv-finalproject-hackathon-cv-21-lv3/json/test_aac.json" --result_name "submission_mlxcommunity/Llama-3.2-3B-Instruct-4bit_aac" --sample_fp16 True --device cuda:1 # "submission_fp16_aac"

# Step 1: Check if result/evaluation.csv exists
if [ -f result/evaluation.csv ]; then
    echo "[INFO] result/evaluation.csv already exists. Skipping evaluate_salmonn.py execution."
else
    echo "[INFO] Running evaluate_salmonn.py..."
    python /data/audiolm-evaluator/evaluate_salmonn.py \
        --test_ann_path "/data/level4-cv-finalproject-hackathon-cv-21-lv3/json/stage2_test_sep_with_testset_id.json" \
        --result_name mlxcommunity/Llama-3.2-3B-Instruct-4bit_asr --device cuda:1 --sample_fp16 True
        # --result_name evaluation

    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to execute evaluate_salmonn.py. Exiting."
        exit 1
    fi

    if [ ! -f result/evaluation.csv ]; then
        echo "[ERROR] CSV file not created. Exiting."
        exit 1
    fi

    echo "[INFO] evaluate_salmonn.py executed successfully. CSV file created."
fi

# Step 2: Execute evaluate_efficiency_salmonn.py
echo "[INFO] Running evaluate_efficiency_salmonn.py..."
efficiency_metrics=$(python /data/audiolm-evaluator/evaluate_efficiency_salmonn.py --sample_fp16 True)

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to execute evaluate_efficiency_salmonn.py. Exiting."
    exit 1
fi

# Step 3: Execute evaluate_WER_SPIDEr.py
echo "[INFO] Running evaluate_WER_SPIDEr.py..."
evaluation_metrics=$(python /data/audiolm-evaluator/evaluate_WER_SPIDEr.py)

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to execute evaluate_WER_SPIDEr.py. Exiting."
    exit 1
fi

# Step 4: Extract and display metrics from both scripts
# Parsing efficiency metrics with grep, awk, and sed
average_memory_usage=$(echo "$efficiency_metrics" | grep -o '"average_memory_usage_gb":[^,]*' | awk -F':' '{print $2}' | tr -d '[:space:]' | tr -d '}')
average_inference_time=$(echo "$efficiency_metrics" | grep -o '"average_inference_time_sec":[^,]*' | awk -F':' '{print $2}' | tr -d '[:space:]' | tr -d '}')
average_ttft=$(echo "$efficiency_metrics" | grep -o '"average_ttft_sec":[^,]*' | awk -F':' '{print $2}' | tr -d '[:space:]' | tr -d '}')
average_tpot=$(echo "$efficiency_metrics" | grep -o '"average_tpot_sec":[^,]*' | awk -F':' '{print $2}' | tr -d '[:space:]' | tr -d '}')

# Parsing evaluation metrics with grep, awk, and sed
wer=$(echo "$evaluation_metrics" | grep -o '"wer":[^,]*' | awk -F':' '{print $2}' | tr -d '[:space:]' | tr -d '}')
spider=$(echo "$evaluation_metrics" | grep -o '"spider":[^,]*' | awk -F':' '{print $2}' | tr -d '[:space:]' | tr -d '}')

# Handling null or missing values
average_memory_usage=${average_memory_usage:-0}
average_inference_time=${average_inference_time:-0}
average_ttft=${average_ttft:-0}
average_tpot=${average_tpot:-0}
wer=${wer:-0}
spider=${spider:-0}

# Displaying metrics
echo "==================================="
echo "           METRICS SUMMARY         "
echo "==================================="

echo ""
echo -e "💡 \033[1mEfficiency Metrics:\033[0m"
echo "-----------------------------------"
printf "  📌 \033[1mAverage Memory Usage\033[0m : \033[1m%.4f GB\033[0m\n" "$average_memory_usage"
printf "  📌 \033[1mAverage Inference Time\033[0m : \033[1m%.4f seconds\033[0m\n" "$average_inference_time"
printf "  📌 \033[1mAverage TTFT\033[0m           : \033[1m%.4f seconds\033[0m\n" "$average_ttft"
printf "  📌 \033[1mAverage TPOT\033[0m           : \033[1m%.4f seconds\033[0m\n" "$average_tpot"
echo ""

echo -e "💡 \033[1mEvaluation Metrics:\033[0m"
echo "-----------------------------------"
printf "  📌 \033[1mWER\033[0m    : \033[1m%.2f\033[0m\n" "$wer"
printf "  📌 \033[1mSpider\033[0m : \033[1m%.2f\033[0m\n" "$spider"
echo ""

echo "==================================="
echo "         END OF METRICS            "
echo "==================================="