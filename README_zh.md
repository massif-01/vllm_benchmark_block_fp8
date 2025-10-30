# vLLM Block FP8 内核调优工具

> 🌐 [English](README.md) | [中文](README_zh.md)

vLLM 的自动化 Triton w8a8 block FP8 内核调优工具。自动检测模型架构并优化内核配置以获得最佳性能。

## 功能特性

- 🎯 **模型自动检测**: 自动从 HuggingFace 模型配置中提取权重形状
- 🔄 **多 GPU 支持**: 跨多个 GPU 并行调优，更快获得结果
- 📊 **灵活配置**: 支持不同的 TP 大小、块大小和批次大小
- 🚀 **预设脚本**: 流行模型的快速启动脚本（Qwen3、DeepSeek-V3）
- ✅ **环境检查**: 预检查确保环境准备就绪

## 快速开始

### 1. 环境检查

```bash
bash scripts/environment_check.sh
```

### 2. 单模型调优

```bash
# Qwen3-Coder-30B-A3B-Instruct-FP8（优化预设）
bash scripts/tune_qwen3_coder.sh

# Qwen3 模型（自动检测形状）
bash scripts/tune_qwen3.sh Qwen/Qwen3-MoE-A14.5B-Chat 4

# DeepSeek-V3（使用默认形状）
bash scripts/tune_deepseek_v3.sh 8

# 自定义模型（自动检测形状）
bash scripts/tune_custom.sh your-model-name 2
```

### 3. 或直接使用 Python

```bash
# Qwen3-Coder-30B-A3B-Instruct-FP8（使用自动检测）
python benchmark_w8a8_block_fp8.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --tp-size 4 --input-type fp8 --trust-remote-code

# Qwen3 模型（自动检测形状）
python benchmark_w8a8_block_fp8.py --model Qwen/Qwen3-MoE-A14.5B-Chat --tp-size 4 --input-type fp8

# DeepSeek-V3（默认形状）
python benchmark_w8a8_block_fp8.py --tp-size 8 --input-type fp8
```

## 配置说明

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | HuggingFace 模型标识符（自动检测形状） | None（使用 DeepSeek-V3） |
| `--tp-size` | 张量并行大小 | `8` |
| `--input-type` | 输入量化类型 | `fp8` |
| `--out-dtype` | 输出数据类型 | `float16` |
| `--block-n` | 量化的块大小 N | `128` |
| `--block-k` | 量化的块大小 K | `128` |
| `--batch-size` | 要测试的单个批次大小（默认：测试所有） | None |
| `--save-path` | 保存调优配置的目录 | `./tuned_configs` |
| `--trust-remote-code` | 加载模型时信任远程代码 | False |

## 支持的模型

工具自动检测以下模型的权重形状：

- **Qwen3 系列**: Qwen3-MoE、Qwen3-Next 模型
- **DeepSeek-V3**: 内置默认形状
- **通用 Transformer**: 从 `hidden_size` 和 `intermediate_size` 自动检测

对于不支持的模型，工具会回退到 DeepSeek-V3 默认形状。

## 输出

调优后的配置保存为 JSON 文件，格式如下：

```
N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{block_n},{block_k}].json
```

每个文件包含不同批次大小的最优配置：

```json
{
  "1": { "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, ... },
  "64": { "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, ... },
  ...
}
```

### 复制配置到 vLLM

调优后，将生成的配置复制到 vLLM：

```bash
cp tuned_configs/*.json /path/to/vllm/model_executor/layers/quantization/utils/configs/
```

## 使用示例

### 示例 1: 使用 TP=4 调优 Qwen3-Coder-30B-A3B-Instruct-FP8

```bash
bash scripts/tune_qwen3_coder.sh Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 4
```

### 示例 1b: 使用 TP=4 调优 Qwen3-MoE

```bash
bash scripts/tune_qwen3.sh Qwen/Qwen3-MoE-A14.5B-Chat 4
```

### 示例 2: 批量调优多个模型

```bash
bash examples/tune_qwen3_models.sh
```

### 示例 3: 自定义块大小

```bash
python benchmark_w8a8_block_fp8.py \
    --model your-model \
    --tp-size 2 \
    --block-n 64 \
    --block-k 128 \
    --input-type fp8
```

## 项目结构

```
vllm_benchmark_block_fp8/
├── README.md                      # 英文文档
├── README_zh.md                   # 中文文档
├── benchmark_w8a8_block_fp8.py   # 主调优脚本
├── scripts/                       # 辅助脚本
│   ├── environment_check.sh      # 环境检查
│   ├── tune_qwen3_coder.sh      # Qwen3-Coder 预设（已优化）
│   ├── tune_qwen3.sh             # Qwen3 预设
│   ├── tune_deepseek_v3.sh       # DeepSeek-V3 预设
│   └── tune_custom.sh            # 自定义模型预设
├── configs/                       # 配置文件
│   └── model_shapes.json         # 模型形状参考
└── examples/                      # 示例脚本
    └── tune_qwen3_models.sh     # 批量调优示例
```

## 前置要求

- Python 3.8+
- 支持 CUDA 的 PyTorch
- 已安装 vLLM（必须在 Python 路径中）
- 兼容 CUDA 的 GPU
- 必需的 vLLM 模块：`fp8_utils`、`triton_utils`、`transformers_utils`

## 许可证

Apache-2.0 许可证

## 贡献

欢迎贡献！请随时提交 Issues 和 Pull Requests。

---

⭐ **如果这个项目对你有帮助，请给我们一个 Star！**

