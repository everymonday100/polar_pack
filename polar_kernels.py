#!/usr/bin/env python3
import torch
import triton
import triton.language as tl
import math
import time


def pack_amp_phase(amp_q, phase_q):
    packed = ((amp_q.to(torch.int32) & 0x3F) << 10) | (phase_q.to(torch.int32) & 0x3FF)
    return (packed - 32768).to(torch.int16)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 2, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def polar_dual_bitpacked_kernel(
    a_ptr, packed_ptr, bmax_ptr,
    amp_lut_ptr, cos_lut_ptr, sin_lut_ptr,
    out1_ptr, out2_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_pk, stride_pn,
    stride_o1m, stride_o1n,
    stride_o2m, stride_o2n,
    num_phases: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    packed_ptrs = packed_ptr + offs_k[:, None] * stride_pk + offs_bn[None, :] * stride_pn

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        mask_b = offs_k[:, None] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        packed = tl.load(packed_ptrs, mask=mask_b, other=0)

        packed_u = packed.to(tl.int32) + 32768
        amp_q = (packed_u >> 10) & 0x3F
        phase_q = packed_u & 0x3FF

        amp_norm = tl.load(amp_lut_ptr + amp_q, mask=mask_b, other=0.0)

        k_global = k * BLOCK_K + offs_k[:, None]
        group_idx = (k_global * N + offs_bn[None, :]) // group_size
        bmax = tl.load(bmax_ptr + group_idx, mask=mask_b, other=1.0)
        amplitude = amp_norm * bmax.to(tl.float32)

        cos_v = tl.load(cos_lut_ptr + phase_q, mask=mask_b, other=0.0)
        sin_v = tl.load(sin_lut_ptr + phase_q, mask=mask_b, other=0.0)

        b1 = (amplitude * cos_v).to(tl.float16)
        b2 = (amplitude * sin_v).to(tl.float16)

        acc1 += tl.dot(a, b1)
        acc2 += tl.dot(a, b2)

        a_ptrs += BLOCK_K * stride_ak
        packed_ptrs += BLOCK_K * stride_pk

    c1 = acc1.to(tl.float16)
    c2 = acc2.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_out = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(out1_ptr + offs_cm[:, None] * stride_o1m + offs_cn[None, :] * stride_o1n,
             c1, mask=mask_out)
    tl.store(out2_ptr + offs_cm[:, None] * stride_o2m + offs_cn[None, :] * stride_o2n,
             c2, mask=mask_out)


def polar_dual_bitpacked(a_fp16, packed, bmax, amp_lut, cos_lut, sin_lut,
                         num_phases=1024, group_size=32):
    M, K = a_fp16.shape
    K_b, N = packed.shape
    assert K == K_b
    out1 = torch.empty((M, N), dtype=torch.float16, device=a_fp16.device)
    out2 = torch.empty_like(out1)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    polar_dual_bitpacked_kernel[grid](
        a_fp16, packed, bmax, amp_lut, cos_lut, sin_lut, out1, out2,
        M, N, K,
        a_fp16.stride(0), a_fp16.stride(1),
        packed.stride(0), packed.stride(1),
        out1.stride(0), out1.stride(1),
        out2.stride(0), out2.stride(1),
        num_phases, group_size,
    )
    return out1, out2


def quantize_polar_bitpacked(w1, w2, amp_bits=6, num_phases=1024, mu=255, group_size=32):
    K, N = w1.shape
    amplitude = torch.sqrt(w1**2 + w2**2)
    phase = torch.atan2(w2, w1)

    phase_q = (torch.round((phase + math.pi) / (2 * math.pi) * num_phases)
               .long() % num_phases).to(torch.int16)

    total = K * N
    amp_flat = amplitude.flatten()
    pad = (group_size - total % group_size) % group_size
    if pad:
        amp_flat = torch.nn.functional.pad(amp_flat, (0, pad))
    grouped = amp_flat.reshape(-1, group_size)
    bmax = grouped.abs().max(dim=1, keepdim=True).values
    bmax = torch.where(bmax > 1e-10, bmax, torch.ones_like(bmax))
    norm = grouped / bmax

    log_mu = math.log(1 + mu)
    compressed = torch.sign(norm) * torch.log1p(mu * norm.abs()) / log_mu
    levels = (1 << amp_bits) - 1
    amp_q = (torch.round(compressed * levels).clamp(0, levels)
             .to(torch.int8).flatten()[:total].reshape(K, N))
    bmax = bmax.flatten().to(torch.float16)

    packed = pack_amp_phase(amp_q, phase_q)

    phases = torch.linspace(-math.pi, math.pi, num_phases + 1)[:-1]
    cos_lut = torch.cos(phases).to(torch.float16).to(w1.device)
    sin_lut = torch.sin(phases).to(torch.float16).to(w1.device)
    i = torch.arange(levels + 1, dtype=torch.float32)
    amp_lut = ((torch.exp(i / levels * log_mu) - 1.0) / mu).to(torch.float16).to(w1.device)

    return packed, bmax, amp_lut, cos_lut, sin_lut


if __name__ == "__main__":
    K, N = 4096, 4096
    w1 = torch.randn(K, N, dtype=torch.float16, device='cuda')
    w2 = torch.randn(K, N, dtype=torch.float16, device='cuda')
    packed, bmax, amp_lut, cos_lut, sin_lut = quantize_polar_bitpacked(w1, w2)

    print("Bit-packed polar format: 16 bits/weight (6 amplitude + 10 phase)")
    print(f"Memory: packed {packed.element_size() * packed.numel() / 1e6:.1f} MB "
          f"vs fp16 dual {2 * w1.element_size() * w1.numel() / 1e6:.1f} MB")

    print(f"{'M':>5} | {'polar dual':>15} | {'fp16 dual':>10} | speedup")
    for M in [1, 8, 32, 128, 1024]:
        a = torch.randn(M, K, dtype=torch.float16, device='cuda')
        for _ in range(10):
            polar_dual_bitpacked(a, packed, bmax, amp_lut, cos_lut, sin_lut)
            torch.matmul(a, w1); torch.matmul(a, w2)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(100):
            polar_dual_bitpacked(a, packed, bmax, amp_lut, cos_lut, sin_lut)
        torch.cuda.synchronize()
        tp = (time.time() - t0) / 100

        t0 = time.time()
        for _ in range(100):
            torch.matmul(a, w1); torch.matmul(a, w2)
        torch.cuda.synchronize()
        tf = (time.time() - t0) / 100

        print(f"{M:>5} | {tp*1000:>13.2f} ms | {tf*1000:>8.2f} ms | {tf/tp:.2f}x")