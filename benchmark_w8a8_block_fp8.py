# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from sglang quantization/tuning_block_wise_kernel.py
# vLLM W8A8 Block FP8 Kernel Tuning Tool
# vLLM W8A8 Block FP8 内核调优工具

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _w8a8_triton_block_scaled_mm,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.transformers_utils.config import get_config
from vllm.utils.argparse_utils import FlexibleArgumentParser

mp.set_start_method("spawn", force=True)

# Only support CUDA devices / 仅支持 CUDA 设备
assert current_platform.is_cuda(), (
    "Only support tune w8a8 block fp8 kernel on CUDA device."
    "仅支持在 CUDA 设备上调优 w8a8 block fp8 内核"
)

# Data type mapping / 数据类型映射
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "half": torch.half,
    "bfloat16": torch.bfloat16,
}


def w8a8_block_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    config: dict[str, Any],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    This function performs matrix multiplication with block-wise quantization.
    该函数执行块级量化的矩阵乘法
    
    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    它接受两个输入张量 `A` 和 `B`，以及缩放因子 `As` 和 `Bs`
    The output is returned in the specified `output_dtype`.
    输出以指定的 `output_dtype` 返回

    Args:
        A: The input tensor, e.g., activation. / 输入张量，例如激活值
        B: The input tensor, e.g., weight. / 输入张量，例如权重
        As: The per-token-group quantization scale for `A`. / A 的每令牌组量化缩放因子
        Bs: The per-block quantization scale for `B`. / B 的每块量化缩放因子
        block_size: The block size for per-block quantization. / 每块量化的块大小
                    It should be 2-dim, e.g., [128, 128]. / 应该是二维的，例如 [128, 128]
        output_dtype: The dtype of the returned tensor. / 返回张量的数据类型

    Returns:
        torch.Tensor: The result of matmul. / 矩阵乘法的结果
    """
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    if A.dtype == torch.float8_e4m3fn:
        kernel = _w8a8_triton_block_scaled_mm
    else:
        raise RuntimeError("Currently, only support tune w8a8 block fp8 kernel. / 目前仅支持调优 w8a8 block fp8 内核")

    kernel[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return C


def get_configs_compute_bound():
    """
    Generate search space for Triton kernel configurations.
    生成 Triton 内核配置的搜索空间
    
    Returns:
        List of configuration dictionaries / 配置字典列表
    """
    configs = []
    for num_stages in [2, 3, 4, 5]:
        for block_m in [16, 32, 64, 128, 256]:
            for block_k in [64, 128]:
                for block_n in [32, 64, 128, 256]:
                    for num_warps in [4, 8]:
                        for group_size in [1, 16, 32, 64]:
                            configs.append(
                                {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": group_size,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                }
                            )
    return configs


def get_model_weight_shapes(model_name: str, tp_size: int, trust_remote_code: bool = True) -> List[Tuple[int, int]]:
    """
    Automatically detect model architecture and generate weight shapes.
    自动检测模型架构并生成权重形状
    
    Args:
        model_name: HuggingFace model identifier / HuggingFace 模型标识符
        tp_size: Tensor parallelism size / 张量并行大小
        trust_remote_code: Whether to trust remote code / 是否信任远程代码
        
    Returns:
        List of (N, K) weight shape tuples / (N, K) 权重形状元组列表
    """
    try:
        # Load model config / 加载模型配置
        config = get_config(model=model_name, trust_remote_code=trust_remote_code)
        
        # Get model architecture / 获取模型架构
        if hasattr(config, 'architectures') and config.architectures:
            arch = config.architectures[0]
        else:
            arch = "Unknown"
        
        # Extract model dimensions / 提取模型维度
        hidden_size = getattr(config, 'hidden_size', None)
        intermediate_size = getattr(config, 'intermediate_size', None)
        
        # Handle different model architectures / 处理不同的模型架构
        if arch in ("DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"):
            # DeepSeek-V3 specific shapes / DeepSeek-V3 特定形状
            return get_deepseek_v3_shapes(tp_size)
        elif arch in ("Qwen3MoeForCausalLM", "Qwen3NextForCausalLM"):
            # Qwen3 MoE/Next models / Qwen3 MoE/Next 模型
            return get_qwen3_shapes(config, tp_size)
        elif "Qwen" in arch or "qwen" in model_name.lower():
            # Qwen series models / Qwen 系列模型
            return get_qwen_shapes(config, tp_size)
        elif hidden_size and intermediate_size:
            # Generic transformer models / 通用 Transformer 模型
            return get_generic_shapes(hidden_size, intermediate_size, tp_size)
        else:
            print(f"Warning: Cannot auto-detect shapes for {model_name}, using DeepSeek-V3 defaults")
            print(f"警告: 无法自动检测 {model_name} 的形状，使用 DeepSeek-V3 默认值")
            return get_deepseek_v3_shapes(tp_size)
            
    except Exception as e:
        print(f"Error loading model config: {e}")
        print(f"加载模型配置错误: {e}")
        print("Falling back to DeepSeek-V3 default shapes")
        print("回退到 DeepSeek-V3 默认形状")
        return get_deepseek_v3_shapes(tp_size)


def get_deepseek_v3_shapes(tp_size: int) -> List[Tuple[int, int]]:
    """
    Get weight shapes for DeepSeek-V3 model.
    获取 DeepSeek-V3 模型的权重形状
    
    Args:
        tp_size: Tensor parallelism size / 张量并行大小
        
    Returns:
        List of (N, K) weight shape tuples / (N, K) 权重形状元组列表
    """
    # DeepSeek-V3 weight shapes (cannot TP) / DeepSeek-V3 权重形状（不能 TP）
    total = [
        (512 + 64, 7168),
        (2112, 7168),
        ((128 + 64) * 128, 7168),
        (128 * (128 + 128), 512),
        (7168, 16384),
        (7168, 18432),
    ]
    # N can TP / N 可以 TP
    n_tp = [
        (18432 * 2, 7168),
        ((128 + 64) * 128, 7168),
        (128 * (128 + 128), 512),
        (24576, 1536),
        (12288, 7168),
        (4096, 7168),
    ]
    # K can TP / K 可以 TP
    k_tp = [(7168, 18432), (7168, 16384), (7168, 2048)]

    weight_shapes = []
    for t in total:
        weight_shapes.append(t)
    for n_t in n_tp:
        new_t = (n_t[0] // tp_size, n_t[1])
        weight_shapes.append(new_t)
    for k_t in k_tp:
        new_t = (k_t[0], k_t[1] // tp_size)
        weight_shapes.append(new_t)
    return weight_shapes


def get_qwen3_shapes(config, tp_size: int) -> List[Tuple[int, int]]:
    """
    Get weight shapes for Qwen3 series models.
    获取 Qwen3 系列模型的权重形状
    
    Args:
        config: Model configuration object / 模型配置对象
        tp_size: Tensor parallelism size / 张量并行大小
        
    Returns:
        List of (N, K) weight shape tuples / (N, K) 权重形状元组列表
    """
    hidden_size = getattr(config, 'hidden_size', None)
    intermediate_size = getattr(config, 'intermediate_size', None)
    moe_intermediate_size = getattr(config, 'moe_intermediate_size', None)
    
    if not hidden_size:
        print("Warning: Cannot extract hidden_size from Qwen3 config")
        print("警告: 无法从 Qwen3 配置中提取 hidden_size")
        return get_generic_shapes(7168, 18432, tp_size)  # Default to DeepSeek-V3 / 默认使用 DeepSeek-V3
    
    # Qwen3 models use intermediate_size or moe_intermediate_size / Qwen3 模型使用 intermediate_size 或 moe_intermediate_size
    ffn_size = intermediate_size or moe_intermediate_size or (hidden_size * 2)
    
    weight_shapes = []
    
    # QKV projection (N can TP) / QKV 投影（N 可以 TP）
    num_heads = getattr(config, 'num_attention_heads', 32)
    head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
    qkv_dim = num_heads * head_dim * 3
    weight_shapes.append((qkv_dim // tp_size, hidden_size))
    
    # Attention output projection (K can TP) / 注意力输出投影（K 可以 TP）
    weight_shapes.append((hidden_size, hidden_size // tp_size))
    
    # Gate/Up projection (N can TP) / Gate/Up 投影（N 可以 TP）
    weight_shapes.append((ffn_size // tp_size, hidden_size))
    
    # Down projection (K can TP) / Down 投影（K 可以 TP）
    weight_shapes.append((hidden_size, ffn_size // tp_size))
    
    # Add common shapes / 添加通用形状
    weight_shapes.extend([
        (hidden_size, hidden_size),  # Self-attention / 自注意力
        (hidden_size, ffn_size),     # MLP intermediate / MLP 中间层
        (ffn_size, hidden_size),     # MLP output / MLP 输出
    ])
    
    return weight_shapes


def get_qwen_shapes(config, tp_size: int) -> List[Tuple[int, int]]:
    """
    Get weight shapes for Qwen/Qwen2.5 series models.
    获取 Qwen/Qwen2.5 系列模型的权重形状
    
    Args:
        config: Model configuration object / 模型配置对象
        tp_size: Tensor parallelism size / 张量并行大小
        
    Returns:
        List of (N, K) weight shape tuples / (N, K) 权重形状元组列表
    """
    hidden_size = getattr(config, 'hidden_size', 4096)
    intermediate_size = getattr(config, 'intermediate_size', None)
    
    # Qwen2.5 typically has intermediate_size = hidden_size * 4 approximately / Qwen2.5 通常 intermediate_size ≈ hidden_size * 4
    if not intermediate_size:
        intermediate_size = hidden_size * 4
    
    return get_generic_shapes(hidden_size, intermediate_size, tp_size)


def get_generic_shapes(hidden_size: int, intermediate_size: int, tp_size: int) -> List[Tuple[int, int]]:
    """
    Get generic weight shapes for standard transformer models.
    获取标准 Transformer 模型的通用权重形状
    
    Args:
        hidden_size: Hidden dimension size / 隐藏维度大小
        intermediate_size: Intermediate/FFN dimension size / 中间层/FFN 维度大小
        tp_size: Tensor parallelism size / 张量并行大小
        
    Returns:
        List of (N, K) weight shape tuples / (N, K) 权重形状元组列表
    """
    weight_shapes = []
    
    # Standard transformer GEMM shapes / 标准 Transformer GEMM 形状
    # QKV projection (N can TP) / QKV 投影（N 可以 TP）
    weight_shapes.append((hidden_size * 3 // tp_size, hidden_size))
    
    # Attention output (K can TP) / 注意力输出（K 可以 TP）
    weight_shapes.append((hidden_size, hidden_size // tp_size))
    
    # Gate/Up projection (N can TP) / Gate/Up 投影（N 可以 TP）
    weight_shapes.append((intermediate_size // tp_size, hidden_size))
    
    # Down projection (K can TP) / Down 投影（K 可以 TP）
    weight_shapes.append((hidden_size, intermediate_size // tp_size))
    
    # Full shapes (cannot TP) / 完整形状（不能 TP）
    weight_shapes.extend([
        (hidden_size, hidden_size),
        (hidden_size, intermediate_size),
        (intermediate_size, hidden_size),
    ])
    
    return weight_shapes


def benchmark_config(
    A, B, As, Bs, block_size, config, out_dtype=torch.float16, num_iters=10
):
    """
    Benchmark a specific kernel configuration.
    基准测试特定的内核配置
    
    Args:
        A, B, As, Bs: Input tensors and scales / 输入张量和缩放因子
        block_size: Block size for quantization / 量化的块大小
        config: Kernel configuration / 内核配置
        out_dtype: Output data type / 输出数据类型
        num_iters: Number of benchmark iterations / 基准测试迭代次数
        
    Returns:
        Average kernel time in microseconds / 平均内核时间（微秒）
    """
    def run():
        w8a8_block_matmul(A, B, As, Bs, block_size, config, out_dtype)

    torch.cuda.synchronize()
    # JIT compilation & warmup / JIT 编译和预热
    for _ in range(5):
        run()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: list[float] = []
    for i in range(num_iters):
        torch.cuda.synchronize()
        start_event.record()
        run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # Convert to microseconds / 转换为微秒
    return avg


def tune(M, N, K, block_size, out_dtype, search_space, input_type):
    """
    Tune kernel configuration for a specific shape.
    为特定形状调优内核配置
    
    Args:
        M: Batch size / 批次大小
        N: Output dimension / 输出维度
        K: Input dimension / 输入维度
        block_size: Block quantization size / 块量化大小
        out_dtype: Output data type / 输出数据类型
        search_space: List of configurations to test / 要测试的配置列表
        input_type: Input quantization type / 输入量化类型
        
    Returns:
        Best configuration dictionary / 最佳配置字典
    """
    factor_for_scale = 1e-2

    if input_type == "fp8":
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        # Generate random FP8 inputs / 生成随机 FP8 输入
        A_fp32 = (
            (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
        )
        A = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        B_fp32 = (
            (torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
        )
        B = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    else:
        raise RuntimeError("Currently, only support tune w8a8 block fp8 kernel. / 目前仅支持调优 w8a8 block fp8 内核")

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    # Generate quantization scales / 生成量化缩放因子
    As = torch.rand(M, k_tiles, dtype=torch.float32, device="cuda") * factor_for_scale
    Bs = (
        torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="cuda")
        * factor_for_scale
    )

    best_config = None
    best_time = float("inf")
    for config in tqdm(search_space, desc=f"Tuning M={M}, N={N}, K={K}"):
        try:
            kernel_time = benchmark_config(
                A,
                B,
                As,
                Bs,
                block_size,
                config,
                out_dtype,
                num_iters=10,
            )
        except triton.runtime.autotuner.OutOfResources:
            # Some configurations may be invalid and fail to compile. / 某些配置可能无效并编译失败
            continue

        if kernel_time < best_time:
            best_time = kernel_time
            best_config = config
    now = datetime.now()
    print(f"{now.ctime()}] Completed tuning for batch_size={M}, N={N}, K={K}")
    assert best_config is not None
    return best_config


def save_configs(
    N,
    K,
    block_n,
    block_k,
    configs,
    save_path,
    input_type="fp8",
) -> None:
    """
    Save tuned configurations to JSON file.
    将调优后的配置保存到 JSON 文件
    
    Args:
        N: Output dimension / 输出维度
        K: Input dimension / 输入维度
        block_n, block_k: Block sizes / 块大小
        configs: Dictionary mapping batch sizes to configurations / 批次大小到配置的映射字典
        save_path: Directory to save config file / 保存配置文件的目录
        input_type: Input quantization type / 输入量化类型
    """
    os.makedirs(save_path, exist_ok=True)
    device_name = current_platform.get_device_name().replace(" ", "_")
    json_file_name = (
        f"N={N},K={K},device_name={device_name},dtype={input_type}_w8a8,"
        f"block_shape=[{block_n},{block_k}].json"
    )

    config_file_path = os.path.join(save_path, json_file_name)
    print(f"Writing best config to {config_file_path}...")
    print(f"将最佳配置写入 {config_file_path}...")

    with open(config_file_path, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def tune_on_gpu(args_dict):
    """
    Run tuning on a specific GPU.
    在特定 GPU 上运行调优
    
    Args:
        args_dict: Dictionary containing GPU ID, batch sizes, weight shapes, and args / 
                   包含 GPU ID、批次大小、权重形状和参数的字典
    """
    gpu_id = args_dict["gpu_id"]
    batch_sizes = args_dict["batch_sizes"]
    weight_shapes = args_dict["weight_shapes"]
    args = args_dict["args"]

    torch.cuda.set_device(gpu_id)
    print(f"Starting tuning on GPU {gpu_id} with batch sizes {batch_sizes}")
    print(f"在 GPU {gpu_id} 上开始调优，批次大小: {batch_sizes}")

    block_n = args.block_n
    block_k = args.block_k
    out_dtype = DTYPE_MAP[args.out_dtype]
    save_path = args.save_path
    input_type = args.input_type

    search_space = get_configs_compute_bound()
    # Filter configs where BLOCK_SIZE_K divides block_k / 过滤 BLOCK_SIZE_K 能整除 block_k 的配置
    search_space = [
        config for config in search_space if block_k % config["BLOCK_SIZE_K"] == 0
    ]

    start = time.time()
    for shape in tqdm(weight_shapes, desc=f"GPU {gpu_id} - Shapes"):
        N, K = shape[0], shape[1]
        print(f"[GPU {gpu_id}] Tune for weight shape of `N: {N}, K: {K}`")
        benchmark_results = [
            tune(
                batch_size,
                N,
                K,
                [block_n, block_k],
                out_dtype,
                search_space,
                input_type,
            )
            for batch_size in tqdm(batch_sizes, desc=f"GPU {gpu_id} - Batch sizes")
        ]
        best_configs = {M: config for M, config in zip(batch_sizes, benchmark_results)}
        save_configs(N, K, block_n, block_k, best_configs, save_path, input_type)

    end = time.time()
    print(f"Tuning on GPU {gpu_id} took {end - start:.2f} seconds")
    print(f"GPU {gpu_id} 上的调优耗时 {end - start:.2f} 秒")


def distribute_batch_sizes(batch_sizes, num_gpus):
    """
    Distribute batch sizes across available GPUs.
    在可用 GPU 之间分配批次大小
    
    Args:
        batch_sizes: List of batch sizes to test / 要测试的批次大小列表
        num_gpus: Number of available GPUs / 可用 GPU 数量
        
    Returns:
        List of batch size lists, one per GPU / 批次大小列表的列表，每个 GPU 一个
    """
    batches_per_gpu = []
    for i in range(num_gpus):
        start_idx = i * len(batch_sizes) // num_gpus
        end_idx = (i + 1) * len(batch_sizes) // num_gpus
        batches_per_gpu.append(batch_sizes[start_idx:end_idx])
    return batches_per_gpu


def main(args):
    """
    Main tuning function.
    主调优函数
    
    Args:
        args: Parsed command line arguments / 解析后的命令行参数
    """
    print(args)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU available for tuning / 没有可用于调优的 GPU")
    print(f"Found {num_gpus} GPUs for parallel tuning")
    print(f"找到 {num_gpus} 个 GPU 用于并行调优")

    torch.cuda.init()

    # Determine batch sizes to test / 确定要测试的批次大小
    if args.batch_size is None:
        batch_sizes = [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
        ]
    else:
        batch_sizes = [args.batch_size]
        num_gpus = 1  # If only one batch size, use only one GPU / 如果只有一个批次大小，只使用一个 GPU

    # Get weight shapes for the model / 获取模型的权重形状
    if args.model:
        print(f"Auto-detecting weight shapes for model: {args.model}")
        print(f"自动检测模型权重形状: {args.model}")
        weight_shapes = get_model_weight_shapes(args.model, args.tp_size, args.trust_remote_code)
        print(f"Detected {len(weight_shapes)} weight shapes")
        print(f"检测到 {len(weight_shapes)} 个权重形状")
    else:
        # Fallback to DeepSeek-V3 shapes / 回退到 DeepSeek-V3 形状
        print("No model specified, using DeepSeek-V3 default shapes")
        print("未指定模型，使用 DeepSeek-V3 默认形状")
        weight_shapes = get_deepseek_v3_shapes(args.tp_size)

    batches_per_gpu = distribute_batch_sizes(batch_sizes, num_gpus)

    process_args = []
    for gpu_id in range(num_gpus):
        process_args.append(
            {
                "gpu_id": gpu_id,
                "batch_sizes": batches_per_gpu[gpu_id],
                "weight_shapes": weight_shapes,  # Each GPU processes all weight shapes / 每个 GPU 处理所有权重形状
                "args": args,
            }
        )

    ctx = mp.get_context("spawn")
    with ctx.Pool(num_gpus) as pool:
        pool.map(tune_on_gpu, process_args)

    print("Multi-GPU tuning completed")
    print("多 GPU 调优完成")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="""
Tune triton w8a8 block fp8 kernel for various models.
为各种模型调优 triton w8a8 block fp8 内核

Examples / 示例:
    # DeepSeek-V3 / DeepSeek-V3
    python benchmark_w8a8_block_fp8.py --tp-size 8 --input-type fp8
    
    # Qwen3 models with auto-detection / 自动检测 Qwen3 模型
    python benchmark_w8a8_block_fp8.py --model Qwen/Qwen3-MoE-A14.5B-Chat --tp-size 4 --input-type fp8
    
    # Custom model / 自定义模型
    python benchmark_w8a8_block_fp8.py --model your-model-name --tp-size 2 --input-type fp8

Then copy generated configs to: model_executor/layers/quantization/utils/configs
然后将生成的配置复制到: model_executor/layers/quantization/utils/configs
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model identifier (auto-detects weight shapes). Leave empty for DeepSeek-V3 defaults. / HuggingFace 模型标识符（自动检测权重形状）。留空则使用 DeepSeek-V3 默认值"
    )
    parser.add_argument("--tp-size", "-tp", type=int, default=8, help="Tensor parallelism size / 张量并行大小")
    parser.add_argument("--input-type", type=str, choices=["fp8"], default="fp8", help="Input quantization type / 输入量化类型")
    parser.add_argument(
        "--out-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16", "half"],
        default="float16",
        help="Output data type / 输出数据类型"
    )
    parser.add_argument("--block-n", type=int, default=128, help="Block size N for quantization / 量化的块大小 N")
    parser.add_argument("--block-k", type=int, default=128, help="Block size K for quantization / 量化的块大小 K")
    parser.add_argument("--batch-size", type=int, required=False, help="Single batch size to test (default: test all sizes) / 要测试的单个批次大小（默认：测试所有大小）")
    parser.add_argument("--save-path", type=str, default="./tuned_configs", help="Directory to save tuned configs / 保存调优配置的目录")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code when loading model config / 加载模型配置时信任远程代码")
    args = parser.parse_args()

    main(args)

