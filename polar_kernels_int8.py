#!/usr/bin/env python3
"""
================================================================
ESPU TRITON KERNELS - Quantized inference на GPU
================================================================
Phase 1, Step 1: Базовый INT8 matmul с fused dequantization
================================================================
"""
import torch
import triton
import triton.language as tl
import time


# ============================================================
# TRITON KERNEL: INT8 Matmul с fused dequantization
# ============================================================
@triton.jit
def int8_matmul_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Scale for dequantization
    scale,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    C = A @ B, где:
    - A: FP16 матрица [M, K] (активации)
    - B: INT8 матрица [K, N] (веса)
    - C: FP16 матрица [M, N] (результат)
    
    Dequantization: B_fp16 = B_int8 * scale
    Fused: dequant происходит в регистрах, без записи обратно в память.
    """
    # Thread block indices
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Compute tile offsets
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers to first blocks
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load A block (FP16)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        
        # Load B block (INT8) и сразу dequantize в FP16
        b_int8 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        b = (b_int8.to(tl.float32) * scale).to(tl.float16)
        
        # Matrix multiply
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Convert to FP16
    c = accumulator.to(tl.float16)
    
    # Write back
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ============================================================
# PYTORCH WRAPPER
# ============================================================
class INT8Matmul(torch.autograd.Function):
    """PyTorch wrapper для Triton kernel."""
    
    @staticmethod
    def forward(ctx, a_fp16, b_int8, scale):
        """
        a_fp16: [M, K] FP16 тензор (активации)
        b_int8: [K, N] INT8 тензор (веса)
        scale: float (для dequantization)
        """
        M, K = a_fp16.shape
        K_b, N = b_int8.shape
        assert K == K_b, f"Dimension mismatch: {K} vs {K_b}"
        
        # Allocate output
        c_fp16 = torch.empty((M, N), dtype=torch.float16, device=a_fp16.device)
        
        # Launch kernel
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )
        
        int8_matmul_kernel[grid](
            a_fp16, b_int8, c_fp16,
            M, N, K,
            a_fp16.stride(0), a_fp16.stride(1),
            b_int8.stride(0), b_int8.stride(1),
            c_fp16.stride(0), c_fp16.stride(1),
            scale,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4
        )
        
        return c_fp16
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass не нужен для inference."""
        raise NotImplementedError("Backward not implemented (inference only)")


def int8_matmul(a_fp16: torch.Tensor, b_int8: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Convenience function для INT8 matmul.
    
    Args:
        a_fp16: [M, K] FP16 тензор (активации)
        b_int8: [K, N] INT8 тензор (веса)
        scale: scale factor для dequantization
    
    Returns:
        [M, N] FP16 тензор (результат)
    """
    return INT8Matmul.apply(a_fp16, b_int8, scale)


# ============================================================
# QUANTIZATION UTILS
# ============================================================
def quantize_to_int8(tensor: torch.Tensor) -> tuple:
    """
    Quantize FP16 tensor to INT8.
    
    Returns:
        (quantized_int8, scale)
    """
    abs_max = tensor.abs().max()
    scale = abs_max / 127.0
    quantized = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale.item()


# ============================================================
# BENCHMARK
# ============================================================
def benchmark_int8_vs_fp16(M=1024, K=4096, N=4096, warmup=10, iters=100):
    """Сравнение скорости INT8 kernel vs PyTorch FP16."""
    print(f"\n🔬 Benchmark: INT8 Triton vs FP16 PyTorch")
    print(f"   Matrix: [{M}, {K}] @ [{K}, {N}]")
    
    # Generate random data
    a_fp16 = torch.randn(M, K, dtype=torch.float16, device='cuda')
    b_fp16 = torch.randn(K, N, dtype=torch.float16, device='cuda')
    b_int8, scale = quantize_to_int8(b_fp16)
    
    # Warmup
    for _ in range(warmup):
        _ = int8_matmul(a_fp16, b_int8, scale)
        _ = torch.matmul(a_fp16, b_fp16)
    torch.cuda.synchronize()
    
    # Benchmark INT8
    start = time.time()
    for _ in range(iters):
        c_int8 = int8_matmul(a_fp16, b_int8, scale)
    torch.cuda.synchronize()
    time_int8 = (time.time() - start) / iters
    
    # Benchmark FP16
    start = time.time()
    for _ in range(iters):
        c_fp16 = torch.matmul(a_fp16, b_fp16)
    torch.cuda.synchronize()
    time_fp16 = (time.time() - start) / iters
    
    # Accuracy check
    error = (c_int8 - c_fp16).abs().mean().item()
    rel_error = error / c_fp16.abs().mean().item()
    
    print(f"\n📊 Results:")
    print(f"   INT8 Triton:  {time_int8*1000:.2f} ms")
    print(f"   FP16 PyTorch: {time_fp16*1000:.2f} ms")
    print(f"   Speedup:      {time_fp16/time_int8:.2f}x")
    print(f"\n🎯 Accuracy:")
    print(f"   Mean abs error:  {error:.6f}")
    print(f"   Relative error:  {rel_error:.6f}")
    
    return {
        'time_int8_ms': time_int8 * 1000,
        'time_fp16_ms': time_fp16 * 1000,
        'speedup': time_fp16 / time_int8,
        'abs_error': error,
        'rel_error': rel_error,
    }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Triton kernels require GPU.")
        exit(1)
    
    # Quick test
    print("🚀 Quick test...")
    a = torch.randn(32, 64, dtype=torch.float16, device='cuda')
    b_fp16 = torch.randn(64, 128, dtype=torch.float16, device='cuda')
    b_int8, scale = quantize_to_int8(b_fp16)
    
    c_triton = int8_matmul(a, b_int8, scale)
    c_torch = torch.matmul(a, b_fp16)
    
    print(f"   Shape: {c_triton.shape}")
    print(f"   Error: {(c_triton - c_torch).abs().mean().item():.6f}")
    
    # Full benchmark
    results = benchmark_int8_vs_fp16()