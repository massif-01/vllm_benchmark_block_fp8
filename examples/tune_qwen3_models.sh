#!/bin/bash

# Example: Batch tuning for multiple Qwen3 models
# 示例：批量调优多个 Qwen3 模型

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)
TUNE_SCRIPT="$PROJECT_DIR/scripts/tune_qwen3.sh"

# List of Qwen3 models to tune / 要调优的 Qwen3 模型列表
MODELS=(
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
    "Qwen/Qwen3-MoE-A14.5B-Chat"
    "Qwen/Qwen3-MoE-A23B-Chat"
    "Qwen/Qwen3-Next-80B-A3B-Instruct"
)

# TP sizes to test / 要测试的 TP 大小
# For 30B model, recommended TP sizes: 4, 8 / 对于 30B 模型，推荐的 TP 大小：4, 8
TP_SIZES=(4 8)

# Block sizes / 块大小
BLOCK_N=128
BLOCK_K=128

echo "Starting batch tuning for ${#MODELS[@]} Qwen3 models..."
echo "开始批量调优 ${#MODELS[@]} 个 Qwen3 模型..."

for model in "${MODELS[@]}"; do
    for tp in "${TP_SIZES[@]}"; do
        echo
        echo "========================================"
        echo "Tuning: $model with TP=$tp"
        echo "调优: $model，TP=$tp"
        echo "========================================"
        
        bash "$TUNE_SCRIPT" "$model" "$tp" "$BLOCK_N" "$BLOCK_K"
        
        if [[ $? -ne 0 ]]; then
            echo "Warning: Tuning failed for $model with TP=$tp"
            echo "警告: $model TP=$tp 的调优失败"
        fi
        
        # Wait a bit between runs / 运行之间稍等片刻
        sleep 5
    done
done

echo
echo "Batch tuning completed!"
echo "批量调优完成！"

