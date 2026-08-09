#!/usr/bin/env python3
"""
Quad-pack Triton decode kernel: 1 memory pass -> 4 matmuls.
uint32 word: [norm 8][t1 8][t2 8][t3 8]
"""
import torch
import triton
import triton.language as tl
import math
import time
import numpy as np

MU = 255.0


def pack_quad_layer(w1, w2, w3, w4, mu=MU):
    """Pack 4 weight matrices (K,N) into uint32 codes + LUTs."""
    ws = [w1, w2, w3, w4]
    s = [torch.sqrt((w.float()**2).mean()) for w in ws]
    wn = [w.float() / sk for w, sk in zip(ws, s)]

    r = torch.sqrt(sum(x**2 for x in wn))
    r_max = float(r.max()) * 1.001
    log_mu = math.log1p(mu)

    t1 = torch.atan2(torch.sqrt(wn[1]**2 + wn[2]**2 + wn[3]**2), wn[0])
    t2 = torch.atan2(torch.sqrt(wn[2]**2 + wn[3]**2), wn[1])
    t3 = torch.atan2(wn[3], wn[2]) % (2 * math.pi)

    nq = torch.round(torch.log1p(mu * r / r_max) / log_mu * 255).clip(0, 255)
    q1 = torch.round(t1 / math.pi * 255).clip(0, 255)
    q2 = torch.round(t2 / math.pi * 255).clip(0, 255)
    q3 = torch.round(t3 / (2 * math.pi) * 255).clip(0, 255)

    to_u32 = lambda t: t.to(torch.int64).cpu().numpy().astype(np.uint32)
    codes_np = ((to_u32(nq) << 24) | (to_u32(q1) << 16) | (to_u32(q2) << 8) | to_u32(q3)).astype(np.uint32)
    codes = torch.from_numpy(codes_np).to(w1.device)

    i = torch.arange(256, dtype=torch.float32)
    norm_lut = ((torch.exp(i / 255 * log_mu) - 1) * r_max / mu).to(torch.float16)
    cos1 = torch.cos(i / 255 * math.pi).to(torch.float16)
    sin1 = torch.sin(i / 255 * math.pi).to(torch.float16)
    cos2 = torch.cos(i / 255 * math.pi).to(torch.float16)
    sin2 = torch.sin(i / 255 * math.pi).to(torch.float16)
    cos3 = torch.cos(i / 255 * 2 * math.pi).to(torch.float16)
    sin3 = torch.sin(i / 255 * 2 * math.pi).to(torch.float16)

    luts = [t.to(w1.device) for t in (norm_lut, cos1, sin1, cos2, sin2, cos3, sin3)]
    scales = [float(sk) for sk in s]
    return codes, luts, scales


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
def quad_bitpacked_kernel(
    a_ptr, codes_ptr,
    norm_lut_ptr, cos1_ptr, sin1_ptr, cos2_ptr, sin2_ptr, cos3_ptr, sin3_ptr,
    out1_ptr, out2_ptr, out3_ptr, out4_ptr,
    s0, s1, s2, s3,
    M, N, K,
    stride_am, stride_ak,
    stride_ck, stride_cn,
    stride_om, stride_on,
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
    c_ptrs = codes_ptr + offs_k[:, None] * stride_ck + offs_bn[None, :] * stride_cn

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        mask_b = offs_k[:, None] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        codes = tl.load(c_ptrs, mask=mask_b, other=0)

        nq = (codes >> 24) & 0xFF
        q1 = (codes >> 16) & 0xFF
        q2 = (codes >> 8) & 0xFF
        q3 = codes & 0xFF

        r = tl.load(norm_lut_ptr + nq, mask=mask_b, other=0.0)
        c1 = tl.load(cos1_ptr + q1, mask=mask_b, other=0.0)
        s1v = tl.load(sin1_ptr + q1, mask=mask_b, other=0.0)
        c2 = tl.load(cos2_ptr + q2, mask=mask_b, other=0.0)
        s2v = tl.load(sin2_ptr + q2, mask=mask_b, other=0.0)
        c3 = tl.load(cos3_ptr + q3, mask=mask_b, other=0.0)
        s3v = tl.load(sin3_ptr + q3, mask=mask_b, other=0.0)

        rs = r * s1v
        rss = rs * s2v
        w1 = (r * c1).to(tl.float16)
        w2 = (rs * c2).to(tl.float16)
        w3 = (rss * c3).to(tl.float16)
        w4 = (rss * s3v).to(tl.float16)

        acc1 += tl.dot(a, w1)
        acc2 += tl.dot(a, w2)
        acc3 += tl.dot(a, w3)
        acc4 += tl.dot(a, w4)

        a_ptrs += BLOCK_K * stride_ak
        c_ptrs += BLOCK_K * stride_ck

    o1 = (acc1 * s0).to(tl.float16)
    o2 = (acc2 * s1).to(tl.float16)
    o3 = (acc3 * s2).to(tl.float16)
    o4 = (acc4 * s3).to(tl.float16)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_out = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    base = offs_cm[:, None] * stride_om + offs_cn[None, :] * stride_on
    tl.store(out1_ptr + base, o1, mask=mask_out)
    tl.store(out2_ptr + base, o2, mask=mask_out)
    tl.store(out3_ptr + base, o3, mask=mask_out)
    tl.store(out4_ptr + base, o4, mask=mask_out)


def quad_bitpacked(a, codes, luts, scales):
    M, K = a.shape
    K_b, N = codes.shape
    assert K == K_b
    outs = [torch.empty((M, N), dtype=torch.float16, device=a.device) for _ in range(4)]
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    quad_bitpacked_kernel[grid](
        a, codes, *luts, *outs, *scales,
        M, N, K,
        a.stride(0), a.stride(1),
        codes.stride(0), codes.stride(1),
        outs[0].stride(0), outs[0].stride(1),
    )
    return outs


if __name__ == "__main__":
    K, N = 4096, 4096
    dev = 'cuda'
    ws = [torch.randn(K, N, dtype=torch.float16, device=dev) for _ in range(4)]
    codes, luts, scales = pack_quad_layer(*ws)

    print("Quad kernel: 32 bits/tuple -> 4 models, 8 bits/weight")
    print(f"Memory: codes {codes.element_size() * codes.numel() / 1e6:.1f} MB "
          f"vs fp16 x4 {4 * ws[0].element_size() * ws[0].numel() / 1e6:.1f} MB")

    # Correctness: kernel vs CPU dequant matmul
    a = torch.randn(16, K, dtype=torch.float16, device=dev)
    outs = quad_bitpacked(a, codes, luts, scales)
    print(f"Correctness: max |diff| vs fp16 matmul = "
          f"{max(float((outs[k] - a @ ws[k]).abs().max()) for k in range(4)):.4f}")

    print(f"{'M':>5} | {'quad x4':>12} | {'fp16 x4':>10} | speedup")
    for M in [1, 8, 32, 128, 1024]:
        a = torch.randn(M, K, dtype=torch.float16, device=dev)
        for _ in range(10):
            quad_bitpacked(a, codes, luts, scales)
            for w in ws:
                torch.matmul(a, w)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(100):
            quad_bitpacked(a, codes, luts, scales)
        torch.cuda.synchronize()
        tq = (time.time() - t0) / 100

        t0 = time.time()
        for _ in range(100):
            for w in ws:
                torch.matmul(a, w)
        torch.cuda.synchronize()
        tf = (time.time() - t0) / 100

        print(f"{M:>5} | {tq*1000:>10.2f} ms | {tf*1000:>8.2f} ms | {tf/tq:.2f}x")