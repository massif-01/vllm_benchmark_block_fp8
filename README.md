# vLLM Block FP8 Kernel Tuning Tool

> 🌐 [English](README.md) | [中文](README_zh.md)

Automated Triton w8a8 block FP8 kernel tuning tool for vLLM. Automatically detects model architecture and optimizes kernel configurations for maximum performance.

## Features

- 🎯 **Model Auto-Detection**: Automatically extracts weight shapes from HuggingFace model configs
- 🔄 **Multi-GPU Support**: Parallel tuning across multiple GPUs for faster results
- 📊 **Flexible Configuration**: Support for different TP sizes, block sizes, and batch sizes
- 🚀 **Preset Scripts**: Quick-start scripts for popular models (Qwen3, DeepSeek-V3)
- ✅ **Environment Check**: Pre-flight checks to ensure environment readiness
- 🔢 **Multiple Quantization Methods**: Support for FP8 (W8A8), INT8 (W8A8), and AWQ (W4A16)

## Quick Start

### 1. Environment Check

```bash
bash scripts/environment_check.sh
```

### 2. Single Model Tuning

```bash
# Qwen3-Coder-30B-A3B-Instruct-FP8 (optimized preset)
bash scripts/tune_qwen3_coder.sh

# Qwen3 models (auto-detects shapes)
bash scripts/tune_qwen3.sh Qwen/Qwen3-MoE-A14.5B-Chat 4

# DeepSeek-V3 (uses default shapes)
bash scripts/tune_deepseek_v3.sh 8

# Custom model (auto-detects shapes)
bash scripts/tune_custom.sh your-model-name 2
```

### 3. Or Use Python Directly

```bash
# Qwen3-Coder-30B-A3B-Instruct-FP8 (with auto-detection)
python benchmark_w8a8_block_fp8.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --tp-size 4 --input-type fp8 --trust-remote-code

# Qwen3 models (auto-detects shapes)
python benchmark_w8a8_block_fp8.py --model Qwen/Qwen3-MoE-A14.5B-Chat --tp-size 4 --input-type fp8

# DeepSeek-V3 (default shapes)
python benchmark_w8a8_block_fp8.py --tp-size 8 --input-type fp8
```

## Configuration

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | HuggingFace model identifier (auto-detects shapes) | None (uses DeepSeek-V3) |
| `--tp-size` | Tensor parallelism size | `8` |
| `--input-type` | Input quantization type | `fp8` |
| `--out-dtype` | Output data type | `float16` |
| `--block-n` | Block size N for quantization | `128` |
| `--block-k` | Block size K for quantization | `128` |
| `--batch-size` | Single batch size to test (default: test all) | None |
| `--save-path` | Directory to save tuned configs | `./tuned_configs` |
| `--trust-remote-code` | Trust remote code when loading model | False |

## Supported Models

The tool automatically detects weight shapes for:

- **Qwen3 Series**: Qwen3-MoE, Qwen3-Next, Qwen3-Coder, Qwen3-VL, Qwen3-Omni models
- **DeepSeek-V3**: Default shapes built-in
- **Generic Transformers**: Auto-detects from `hidden_size` and `intermediate_size`

For unsupported models, the tool falls back to DeepSeek-V3 default shapes.

## Available Tuning Scripts

### 1. W8A8 Block FP8 (`benchmark_w8a8_block_fp8.py`)
- **Quantization**: FP8 weights and activations (W8A8)
- **Use Case**: High-performance inference with FP8-capable GPUs (Ada/Hopper)
- **Requirements**: GPU compute capability >= 8.9

### 2. W8A8 Block INT8 (`benchmark_w8a8_block_int8.py`)
- **Quantization**: INT8 weights and activations (W8A8)
- **Use Case**: Memory-efficient inference on GPUs with compute capability >= 7.5
- **Requirements**: NVIDIA GPU (Turing/Ampere/Ada/Hopper)

### 3. AWQ W4A16 (`benchmark_awq_w4a16.py`)
- **Quantization**: 4-bit weights, 16-bit activations (W4A16)
- **Use Case**: Maximum memory savings with good performance
- **Requirements**: GPU compute capability >= 7.5
- **Documentation**: See [README_AWQ.md](README_AWQ.md) for detailed usage

### Model-Specific Scripts

- `benchmark_w8a8_block_fp8_qwen3_30b.py`: Optimized for Qwen3-30B-A3B-Instruct-2507-Int8-W8A16
- `benchmark_w8a8_block_fp8_qwen3omni_talker.py`: Optimized for Qwen3-Omni-30B-A3B-Instruct (talker config)

## Output

Tuned configurations are saved as JSON files in the format:

```
N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{block_n},{block_k}].json
```

Each file contains optimal configurations for different batch sizes:

```json
{
  "1": { "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, ... },
  "64": { "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, ... },
  ...
}
```

### Copying Configs to vLLM

After tuning, copy the generated configs to vLLM:

```bash
cp tuned_configs/*.json /path/to/vllm/model_executor/layers/quantization/utils/configs/
```

## Examples

### Example 1: Tune Qwen3-Coder-30B-A3B-Instruct-FP8 with TP=4

```bash
bash scripts/tune_qwen3_coder.sh Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 4
```

### Example 1b: Tune Qwen3-MoE with TP=4

```bash
bash scripts/tune_qwen3.sh Qwen/Qwen3-MoE-A14.5B-Chat 4
```

### Example 2: Batch Tune Multiple Models

```bash
bash examples/tune_qwen3_models.sh
```

### Example 3: Custom Block Sizes

```bash
python benchmark_w8a8_block_fp8.py \
    --model your-model \
    --tp-size 2 \
    --block-n 64 \
    --block-k 128 \
    --input-type fp8
```

### Example 4: AWQ W4A16 Tuning

```bash
# Basic AWQ tuning
python benchmark_awq_w4a16.py \
    --tp-size 1 \
    --group-size 128 \
    --split-k-iters 1 \
    --save-path ./awq_configs

# See README_AWQ.md for detailed AWQ usage
```

## Project Structure

```
vllm_benchmark_block_fp8/
├── README.md                              # English documentation
├── README_zh.md                           # Chinese documentation
├── README_AWQ.md                          # AWQ W4A16 tuning documentation
├── benchmark_w8a8_block_fp8.py            # W8A8 Block FP8 tuning script
├── benchmark_w8a8_block_int8.py            # W8A8 Block INT8 tuning script
├── benchmark_awq_w4a16.py                 # AWQ W4A16 tuning script
├── benchmark_w8a8_block_fp8_qwen3_30b.py  # Qwen3-30B optimized script
├── benchmark_w8a8_block_fp8_qwen3omni_talker.py  # Qwen3-Omni Talker optimized script
├── scripts/                               # Helper scripts
│   ├── environment_check.sh              # Environment check
│   ├── tune_qwen3_coder.sh              # Qwen3-Coder preset (optimized)
│   ├── tune_qwen3.sh                     # Qwen3 preset
│   ├── tune_deepseek_v3.sh               # DeepSeek-V3 preset
│   └── tune_custom.sh                    # Custom model preset
├── configs/                               # Configuration files
│   └── model_shapes.json                 # Model shape references
└── examples/                              # Example scripts
    └── tune_qwen3_models.sh             # Batch tuning example
```

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support
- vLLM installed (must be in Python path)
- CUDA-compatible GPU(s)
- Required vLLM modules: `fp8_utils`, `triton_utils`, `transformers_utils`

## License

Apache-2.0 License

## Contributing

Contributions are welcome! Please feel free to submit Issues and Pull Requests.

---

⭐ **If this project helps you, please give us a Star!**

