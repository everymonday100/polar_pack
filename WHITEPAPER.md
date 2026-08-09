# Polar & Quaternion Multi-Model Packing
## Near-Lossless Weight Compression via 2D/4D Spherical Quantization

**Version:** 2.0 (quad-pack release)
**Date:** 2026-08-09

---

## Abstract

We present two related formats for packing multiple LLMs of compatible
architecture into a single weight stream:

- **Dual-pack (`.dualpack`)**: 2 models -> complex plane, polar quantization,
  16 bits/pair (8 bits/model), 1.76-1.78x compression vs FP16.
- **Quad-pack (`.quadpack`)**: 4 models -> quaternion / 4D hypersphere,
  32 bits/tuple (8 bits/model), 1.74x compression vs 4xFP16, with
  **statistically lossless** quality (PPL degradation ~ 0.0%) and a Triton
  decode kernel **1.6-2.0x faster than FP16**.

The key insight: fine-tuned models of a shared base are highly correlated,
so their joint weight distribution concentrates near a low-dimensional
manifold. Spherical coding in 2D/4D exploits this structure, and higher
dimensionality (4D) is information-theoretically *more* efficient, not less.

---

## 1. Introduction

Serving multiple specialist LLMs (instruct, code, math, translation) normally
requires N separate FP16 weight sets. We observe that:

1. Fine-tunes of a common base share most of their information; per-weight
   differences are small residuals.
2. A tuple of N corresponding weights (w1..wN) can be treated as a vector in
   R^N and coded in spherical coordinates (norm + direction).
3. Vector quantization in higher dimension approaches the Shannon
   rate-distortion bound more closely, so 4D coding beats 2D at equal rate.

We pack N models by quantizing each N-tuple once, storing a single coded
word, and reconstructing each model by "looking at its own angle."

---

## 2. Dual-Model Packing (Complex / 2D Polar)

### 2.1 Representation

For a pair (w1, w2) define z = w1 + i*w2 = |z| * e^(i*phi):

    |z| = sqrt(w1^2 + w2^2)      phi = atan2(w2, w1)
    w1 = |z|*cos(phi)            w2 = |z|*sin(phi)

### 2.2 Quantization

- **Amplitude**: 6-bit mu-law (mu=255) with per-group FP16 scale (group=32).
- **Phase**: 10-bit uniform on [-pi, pi).
- **Bit-pack**: `amp(6) << 10 | phase(10)` into uint16.
- **Plug table**: top ~3% outliers (by amplitude) stored as exact FP16 pairs.

Effective rate: 16 bits/pair + overhead ~= **8.97 bits/weight**.

### 2.3 File layout (`.dualpack`)

    [header][codes u16 x n_pairs][scales f16 x n_groups]
    [plug_idx u32 x n][plug_vals f16 x 2n]

Header stores n1, n2 (independent lengths), mu, group size, plug count.

---

## 3. Quad-Model Packing (Quaternion / 4D Spherical)

### 3.1 Why 4D

A 4-tuple (w1..w4) -> quaternion q = w1 + i*w2 + j*w3 + k*w4.
We do **not** use quaternion multiplication (non-commutativity is
irrelevant); the quaternion is a *container* for 4 scalars, coded in
hyperspherical coordinates:

    r  = |q|                                  (norm)
    t1 = atan2(sqrt(w2^2+w3^2+w4^2), w1)      in [0, pi]
    t2 = atan2(sqrt(w3^2+w4^2), w2)           in [0, pi]
    t3 = atan2(w4, w3) mod 2*pi               in [0, 2*pi)

    w1 = r*cos(t1)
    w2 = r*sin(t1)*cos(t2)
    w3 = r*sin(t1)*sin(t2)*cos(t3)
    w4 = r*sin(t1)*sin(t2)*sin(t3)

### 3.2 Bit allocation: 8 + 24 (byte-aligned)

- **Norm**: 8-bit mu-law (global r_max + per-model RMS scale).
- **Angles**: 8 bits each (t1, t2, t3) -> 256-entry LUTs.
- **Word**: uint32 `[norm 8][t1 8][t2 8][t3 8]`, bits 31-24/23-16/15-8/7-0.

Byte alignment is a deliberate kernel feature: one uint32 load, unpack via
`>>24, >>16, >>8, &0xFF`, seven 256-entry LUTs (~3.5 KB, fits L1).

### 3.3 Per-model scale and tails

- Each model is normalized by its global RMS before coding; the 4 RMS values
  are stored in the header (restored = dequant * RMS_k).
- Models longer than n_min store their tail as raw FP16 (v1.1), so every
  weight of every model is recovered (e.g. Ministral lm_head).

### 3.4 Plug table

Top ~5% tuples (by relative reconstruction error) stored as exact FP16
4-tuples. This is our random-access substitute for entropy coding: bits are
spent only on the heavy tail.

### 3.5 File layout (`.quadpack` v2)

    [header: magic, ver, n_models, n_min, bits, mu, plug_n, RMS x4, r_max, tails x4]
    [codes u32 x n_min]
    [plug_idx u32 x n][plug_vals f16 x 4n]
    [tail_0..3 f16]

---

## 4. Rate-Distortion Analysis

### 4.1 Shannon bound

For a Gaussian source, D(R) = sigma^2 * 2^(-2R). At R = 8 bits/weight:
D(8) ~= 1.5e-5 relative MSE (cosine ~= 0.999992). This is a lower bound
achievable only by optimal vector quantizers.

### 4.2 Measured gap

| Format | Bits/weight | RelMSE | Gap to bound |
|--------|------------|--------|--------------|
| Dual (2D polar) | 8.0 | ~4e-4 | ~26x |
| Quad (4D spherical) | 8.48 | ~9e-5 | ~6x |

Quad is **4.4x better** than dual at equal rate: 4D vector quantization is
closer to the bound (shaping gain), and the norm receives 8 bits instead of 6.

### 4.3 Correlation structure ("the diagonal")

After per-model RMS normalization, the 4-tuple direction is **not** uniform
on S3: it concentrates in a cap around the mean direction (models share a
base). Measured: direction component variance 0.168 x4 (sum 0.67 < 1.0),
norm/sigma = 1.844 vs chi-4 expectation 1.88. This lowers direction entropy
- a future delta-coding stage could exploit it, but uniform spherical
coding already wins.

### 4.4 Bit-allocation sweep (sample, 100M weights)

| Alloc | Plug | Bits/W | Cosine | RelMSE |
|-------|------|--------|--------|--------|
| 6+26 | 0.03 | 8.48 | 0.999600 | 8.0e-4 |
| 7+25 | 0.03 | 8.48 | 0.999890 | 2.2e-4 |
| 8+24 | 0.03 | 8.48 | 0.999954 | 9.1e-5 |

Full-data validation (3.09B weights, 8+24/plug 0.08): cosine **0.999958**,
relMSE **8.49e-5** - sample prediction matched to 4 decimal places
(concentration of measure).

---

## 5. GPU Kernels (Triton)

### 5.1 Dual kernel

Dequant-in-kernel: uint16 load -> unpack -> 3 LUT gathers -> 2 dots.
Best: **0.91x FP16** at M=1, 0.92x at M=32 (memory-bound; 2x less traffic).

### 5.2 Quad kernel

One memory pass yields **4 matmuls**: uint32 load -> byte-unpack -> 7 LUT
gathers -> 4 dots; per-model RMS applied to accumulators at store.
Traffic: 67 MB vs 134 MB for 4xFP16 (4096x4096 layer).

### 5.3 Benchmarks (RTX 4060, 4096x4096)

| M | Quad x4 | FP16 x4 | Speedup |
|---|---------|---------|---------|
| 1 | 0.34 ms | 0.55 ms | **1.58x** |
| 8 | 0.36 ms | 0.57 ms | **1.59x** |
| 32 | 0.31 ms | 0.63 ms | **2.00x** |
| 128 | 1.41 ms | 0.63 ms | 0.44x |
| 1024 | 10.67 ms | 4.56 ms | 0.43x |

Decode (memory-bound) beats FP16; prefill (compute-bound) loses to cuBLAS -
inherent to dequant-in-kernel formats.

---

## 6. Experiments

### 6.1 Models

- Dual: Qwen2.5-3B-Instruct + Qwen2.5-Coder-3B-Instruct;
  Ministral-3B-Instruct + Qwen2.5-Coder-3B-Instruct.
- Quad: Qwen2.5-3B-Instruct, Qwen2.5-Coder-3B-Instruct, Qwen2.5-3B (base),
  Ministral-3B-Instruct (3.09B shared weights + Ministral tail).

### 6.2 Dual results

| Pair | Compression | Bits/W | Cosine | PPL delta |
|------|------------|--------|--------|-------|
| Qwen+Qwen | 1.78x | 8.97 | 0.9998 | <0.6% |
| Ministral+Qwen | 1.76x | 9.11 | 0.9996/0.9998 | 0.31% avg |

### 6.3 Quad results

| Metric | Value |
|--------|-------|
| Compression vs 4xFP16 | 1.74x (24.7 GB -> 14.4 GB) |
| Cosine | 0.999958 |
| PPL delta (per model) | -0.008 / -0.030 / -0.045 / +0.074 % |
| **Avg PPL delta** | **-0.002% (statistical lossless)** |
| Decode speedup | 1.6-2.0x |

---

## 7. Limitations

- Models must share tensor layout (identical Linear shapes); tails handle
  length mismatch only.
- Kernels are research prototypes (Triton), not production cuBLAS.
- Prefill slower than FP16 (dequant ALU cost).
- Near-lossless, not bit-exact (plug covers the tail; PPL delta ~ 0).

## 8. Future Work

- [ ] GGUF / llama.cpp port (Q_POLAR / Q_QUAD block types).
- [ ] Group-reference spherical coding (aligned delta, v2).
- [ ] Quantization-aware training in polar/quad space.
- [ ] Fused CUDA decode kernel.

---

*Entropy coding is deliberately avoided: variable-length codes break GPU
random access. Fixed-width spherical codes + plug table is the chosen
rate/distortion/parallelism trade-off.*