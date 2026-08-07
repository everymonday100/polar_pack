#!/usr/bin/env python3
import torch
import triton
import triton.language as tl
import time


@triton.jit
def int8_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b_int8 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        b = (b_int8.to(tl.float32) * scale).to(tl.float16)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


class INT8Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_fp16, b_int8, scale):
        M, K = a_fp16.shape
        K_b, N = b_int8.shape
        assert K == K_b
        c_fp16 = torch.empty((M, N), dtype=torch.float16, device=a_fp16.device)
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
        raise NotImplementedError


def int8_matmul(a_fp16, b_int8, scale):
    return INT8Matmul.apply(a_fp16, b_int8, scale)


def quantize_to_int8(tensor):
    abs_max = tensor.abs().max()
    scale = abs_max / 127.0
    quantized = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale.item()


def benchmark_int8_vs_fp16(M=1024, K=4096, N=4096, warmup=10, iters=100):
    print(f"Benchmark: INT8 Triton vs FP16 PyTorch")
    print(f"Matrix: [{M}, {K}] @ [{K}, {N}]")

    a_fp16 = torch.randn(M, K, dtype=torch.float16, device='cuda')
    b_fp16 = torch.randn(K, N, dtype=torch.float16, device='cuda')
    b_int8, scale = quantize_to_int8(b_fp16)

    for _ in range(warmup):
        _ = int8_matmul(a_fp16, b_int8, scale)
        _ = torch.matmul(a_fp16, b_fp16)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        int8_matmul(a_fp16, b_int8, scale)
    torch.cuda.synchronize()
    time_int8 = (time.time() - start) / iters

    start = time.time()
    for _ in range(iters):
        torch.matmul(a_fp16, b_fp16)
    torch.cuda.synchronize()
    time_fp16 = (time.time() - start) / iters

    c_int8 = int8_matmul(a_fp16, b_int8, scale)
    c_fp16 = torch.matmul(a_fp16, b_fp16)
    error = (c_int8 - c_fp16).abs().mean().item()
    rel_error = error / c_fp16.abs().mean().item()

    print(f"INT8 Triton:  {time_int8*1000:.2f} ms")
    print(f"FP16 PyTorch: {time_fp16*1000:.2f} ms")
    print(f"Speedup:      {time_fp16/time_int8:.2f}x")
    print(f"Mean abs error: {error:.6f}")
    print(f"Relative error: {rel_error:.6f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        exit(1)

    a = torch.randn(32, 64, dtype=torch.float16, device='cuda')
    b_fp16 = torch.randn(64, 128, dtype=torch.float16, device='cuda')
    b_int8, scale = quantize_to_int8(b_fp16)

    c_triton = int8_matmul(a, b_int8, scale)
    c_torch = torch.matmul(a, b_fp16)

    print(f"Shape: {c_triton.shape}")
    print(f"Error: {(c_triton - c_torch).abs().mean().item():.6f}")

    benchmark_int8_vs_fp16()