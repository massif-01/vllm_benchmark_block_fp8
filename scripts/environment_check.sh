#!/bin/bash

# Environment Check Script for Block FP8 Tuning
# Block FP8 调优环境检查脚本
# Checks if the environment is ready for running benchmark_w8a8_block_fp8.py
# 检查环境是否准备好运行 benchmark_w8a8_block_fp8.py

echo "==================== ENVIRONMENT CHECK ===================="
echo "Checking prerequisites for w8a8 block fp8 kernel tuning..."
echo "检查 w8a8 block fp8 内核调优的前置条件..."
echo

ERRORS=0
WARNINGS=0

# Check Python environment / 检查 Python 环境
echo "[1/6] Checking Python environment..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "✓ Python found: $PYTHON_VERSION"
else
    echo "✗ ERROR: Python not found"
    ((ERRORS++))
fi
echo

# Check PyTorch and CUDA / 检查 PyTorch 和 CUDA
echo "[2/6] Checking PyTorch and CUDA..."
if python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null; then
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [[ "$CUDA_AVAILABLE" == "True" ]]; then
        NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        echo "✓ CUDA available: $NUM_GPUS GPU(s)"
    else
        echo "✗ ERROR: CUDA not available. This script requires CUDA GPU."
        echo "  错误: CUDA 不可用。此脚本需要 CUDA GPU"
        ((ERRORS++))
    fi
else
    echo "✗ ERROR: PyTorch not installed or cannot import"
    echo "  错误: PyTorch 未安装或无法导入"
    ((ERRORS++))
fi
echo

# Check vLLM installation / 检查 vLLM 安装
echo "[3/6] Checking vLLM installation..."
if python -c "import vllm" 2>/dev/null; then
    VLLM_VERSION=$(python -c "import vllm; print(getattr(vllm, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
    echo "✓ vLLM found: $VLLM_VERSION"
else
    echo "✗ ERROR: vLLM not installed. Install with: pip install vllm"
    echo "  错误: vLLM 未安装。安装命令: pip install vllm"
    ((ERRORS++))
fi
echo

# Check required vLLM modules / 检查必需的 vLLM 模块
echo "[4/6] Checking required vLLM modules..."
REQUIRED_MODULES=(
    "vllm.model_executor.layers.quantization.utils.fp8_utils"
    "vllm.triton_utils"
    "vllm.transformers_utils.config"
)
for module in "${REQUIRED_MODULES[@]}"; do
    if python -c "import $module" 2>/dev/null; then
        echo "✓ $module"
    else
        echo "✗ ERROR: $module not found"
        ((ERRORS++))
    fi
done
echo

# Check CUDA device availability / 检查 CUDA 设备可用性
echo "[5/6] Checking CUDA devices..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [[ $GPU_COUNT -gt 0 ]]; then
        echo "✓ GPU detected: $GPU_COUNT GPU(s)"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/^/  /'
    else
        echo "⚠ WARNING: No GPUs detected via nvidia-smi"
        ((WARNINGS++))
    fi
else
    echo "⚠ WARNING: nvidia-smi not found. Cannot verify GPU status."
    ((WARNINGS++))
fi
echo

# Check disk space / 检查磁盘空间
echo "[6/6] Checking disk space..."
AVAILABLE_SPACE=$(df -h "$(pwd)" | awk 'NR==2 {print $4}')
echo "✓ Available disk space: $AVAILABLE_SPACE"
if [[ $(df "$(pwd)" | awk 'NR==2 {print $4}') -lt 5242880 ]]; then
    echo "⚠ WARNING: Less than 5GB available. Tuning may require significant space."
    echo "  警告: 可用空间少于 5GB。调优可能需要大量空间"
    ((WARNINGS++))
fi
echo

# Summary / 摘要
echo "==================== CHECK SUMMARY ===================="
if [[ $ERRORS -eq 0 ]]; then
    echo "✓ All critical checks passed!"
    echo "✓ 所有关键检查通过！"
    if [[ $WARNINGS -gt 0 ]]; then
        echo "⚠ $WARNINGS warning(s) found. Review above messages."
        echo "⚠ 发现 $WARNINGS 个警告。请查看上面的消息"
        exit 0
    else
        echo "✓ Environment is ready for kernel tuning!"
        echo "✓ 环境已准备好进行内核调优！"
        exit 0
    fi
else
    echo "✗ $ERRORS error(s) found. Please fix the issues above before running benchmark."
    echo "✗ 发现 $ERRORS 个错误。请在运行基准测试前修复上述问题"
    exit 1
fi

