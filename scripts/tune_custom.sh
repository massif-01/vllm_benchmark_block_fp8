#!/bin/bash

# Custom Model Tuning Script
# 自定义模型调优脚本
# Usage: bash scripts/tune_custom.sh [model_name] [tp_size] [block_n] [block_k]
# 用法: bash scripts/tune_custom.sh [模型名称] [张量并行大小] [块大小N] [块大小K]

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)
BENCHMARK_SCRIPT="$PROJECT_DIR/benchmark_w8a8_block_fp8.py"

# Parse command line arguments / 解析命令行参数
MODEL=${1:-""}      # Model name (required) / 模型名称（必需）
TP=${2:-1}          # Default TP size / 默认 TP 大小
BLOCK_N=${3:-128}   # Default block_n / 默认 block_n
BLOCK_K=${4:-128}   # Default block_k / 默认 block_k

if [[ -z "$MODEL" ]]; then
    echo "Error: Model name is required."
    echo "错误: 需要提供模型名称"
    echo "Usage: $0 <model_name> [tp_size] [block_n] [block_k]"
    exit 1
fi

echo "==================== Custom Model Block FP8 Tuning ===================="
echo "Model: $MODEL"
echo "Tensor Parallelism: $TP"
echo "Block Size: [$BLOCK_N, $BLOCK_K]"
echo "=============================================================="
echo

# Set default values / 设置默认值
INPUT_TYPE=${INPUT_TYPE:-"fp8"}
OUT_DTYPE=${OUT_DTYPE:-"float16"}
SAVE_PATH=${SAVE_PATH:-"./tuned_configs/custom"}

echo "Configuration:"
echo "  MODEL: $MODEL"
echo "  TP_SIZE: $TP"
echo "  BLOCK_N: $BLOCK_N"
echo "  BLOCK_K: $BLOCK_K"
echo "  INPUT_TYPE: $INPUT_TYPE"
echo "  OUT_DTYPE: $OUT_DTYPE"
echo "  SAVE_PATH: $SAVE_PATH"
echo

# Run benchmark script / 运行基准测试脚本
python "$BENCHMARK_SCRIPT" \
    --model "$MODEL" \
    --tp-size "$TP" \
    --input-type "$INPUT_TYPE" \
    --out-dtype "$OUT_DTYPE" \
    --block-n "$BLOCK_N" \
    --block-k "$BLOCK_K" \
    --save-path "$SAVE_PATH" \
    --trust-remote-code

echo
echo "Tuning completed! Configs saved to: $SAVE_PATH"
echo "调优完成！配置已保存到: $SAVE_PATH"
echo "Copy configs to: vllm/model_executor/layers/quantization/utils/configs/"
echo "将配置复制到: vllm/model_executor/layers/quantization/utils/configs/"

