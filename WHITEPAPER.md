```markdown
# Polar Dual-Model Packing

**Two LLMs in one weight stream.** A lossy-but-near-lossless format that stores
two same-architecture models as one bit-packed artifact: 1.78x smaller than two
FP16 models, with both personalities fully restorable and independently usable.

Validated on Qwen2.5-3B-Instruct + Qwen2.5-Coder-3B-Instruct on a consumer
RTX 4060 (8 GB VRAM, 32 GB RAM).

This document is a technical write-up: the idea, the format, the numbers,
and an honest account of what the GPU kernel can and cannot beat.

---

## 1. The problem

Serving two specialized models (e.g. a coding model and a conversational model)
costs 2x memory, 2x storage, 2x download. Standard quantization (GPTQ, AWQ,
GGUF) compresses each model separately - the duplication remains.

Question: can two models be stored and served as **one object**, cheaper than
the sum of its parts?

Observation that makes it possible: weights of two fine-tunes of the same base
are correlated in a loose sense (similar magnitudes, zero-centered
distributions), but more importantly - a pair of real numbers can be treated as
one complex number, and complex numbers have a polar form.

## 2. The idea

For every weight position, take one weight from each model and form

```
z = w1 + i * w2
```

Store it in polar coordinates instead of Cartesian:

- **amplitude** `|z| = sqrt(w1^2 + w2^2)` - non-negative, zero-centered
  distribution, well suited for mu-law companding;
- **phase** `phi = atan2(w2, w1)` - nearly uniform over `[-pi, pi]`,
  well suited for uniform quantization.

Restore is exact in principle:

```
w1 = |z| * cos(phi)
w2 = |z| * sin(phi)
```

Quantization scheme (production parameters):

| Component | Method | Bits |
|---|---|---|
| amplitude | mu-law (mu=255), per-group FP16 scale (group=32) | 6 |
| phase | uniform over 1024 levels | 10 |
| outliers | plug table: top-3% amplitudes stored as exact FP16 | ~0.06 avg |

Total: 16 bits per **pair**, i.e. 8 bits per model, plus small overhead
(scales, plug) - 8.97 bits/weight measured end to end.

Why mu-law for amplitude: weight amplitudes are heavily concentrated near zero
with rare large values; mu-law (the classic G.711 compander) allocates more
levels where the density is higher, without any learned tables. The per-group
FP16 scale handles inter-group magnitude spread; the plug table removes the
residual worst-case outliers that would otherwise dominate the MSE.

Why uniform for phase: empirically the phase histogram of two independent
fine-tunes is close to flat, so 10 uniform bits give ~0.3 degrees of angular
resolution - the dominant error source at 6-bit amplitude stays amplitude,
which is exactly where the bits are spent.

## 3. The format (.dualpack)

Bit packing: `amp(6) << 10 | phase(10)` into one `uint16`. Layout:

```
polar.dualpack
|- header   magic "DUALPACK", version, n1, n2, amp_bits, phase_bits,
|           mu, group_size, chunk_size, plug_n
|- codes    uint16[max(n1, n2)]      bit-packed amp+phase, big-endian per pair
|- scales   fp16[num_groups]         per-group amplitude scale (32 pairs each)
|- plug_i   uint32[plug_n]           sorted indices of outlier pairs
`- plug_v   fp16[plug_n]             exact amplitudes for outliers
```

Measured on the Qwen pair (3,085,697,024 Linear weights per model):

| | Size |
|---|---|
| two FP16 models | 11,771 MB |
| polar.dualpack | 6,599 MB |
| compression | 1.78x |
| effective rate | 8.97 bits/weight |

The shorter model is zero-padded inside the pair stream; padding costs nothing
because zeros quantize to the smallest code.

A JSON sidecar (`*.dualpack.json`) stores method metadata and quality metrics
for tooling and future engine integration (llama.cpp / vLLM).

## 4. Quality

| Metric | Value |
|---|---|
| cosine similarity vs original weights | 0.999801 / 0.999794 |
| round-trip drift vs the FP32 reference pipeline | 0.9999953 |
| perplexity degradation | < 0.6% |

Both models remain fully usable and keep their distinct personalities:
the coder still writes code, the instruct model still explains. This is the
point of the format - it is merging **without identity loss**, unlike weight
averaging where two specialists blur into one generalist.

## 5. Serving: one artifact, three modes

The runtime (`polar_inference.py`, `polar_main.py`) lazily unpacks the artifact
into FP32 weight dumps on first run, then injects them into two HuggingFace
model shells. Three modes:

- **single** - one model answers;
- **parallel** - both answer, for comparison or ensembling;
- **router** - a calibrated heuristic picks one model, or abstains to both.

### Router with soft shifting

The router scores the prompt with weighted regex signals (task verbs x3,
domain concepts x6, inline code x2, weak mentions x1) and calibrates
confidence with Laplace smoothing:

```
confidence = (winner + 1) / (total + 2)
```

so a single weak signal yields 0.67, a 4-vs-0 score yields 0.83, and empty or
tied evidence yields 0.5. Decision policy:

| Condition | Action |
|---|---|
| tie with non-zero evidence | both models |
| mixed-task patterns ("write X, then explain it") | both models |
| confidence < 0.6 | both models |
| confident | one model |
| chosen model emits refusal signals ("I cannot...") | escalation: both |

Per-model generation profiles keep each personality sharp: the coder runs with
`repetition_penalty=1.0` and 512 tokens (code legitimately repeats), the
conversational model with 1.15 and 256.

### Memory management on an 8 GB card

Two 3B FP16 models do not fit in 8 GB VRAM simultaneously, and do not need to.
The manager keeps both in CPU RAM, loads weights lazily on first use, and pins
the active model on GPU with LRU eviction and a 300 s TTL; the resident limit
is auto-detected from VRAM (2 models at >= 15 GB, 1 below). A parallel call on
an 8 GB card therefore costs one PCIe transfer per model switch instead of two
per generation.

The behavior layer is covered by 36/36 unit tests (routing math, abstention
branches, escalation, per-model params, lazy loading, LRU/TTL bookkeeping).

## 6. GPU kernel: what it beats and what it does not

A Triton prototype (`polar_kernels.py`) fuses bit-unpack, mu-law expand (via a
64-entry LUT), phase-to-cos/sin (one 1024-entry LUT, sin obtained by index
shift), per-group rescale and **two** matmuls - both models - in a single pass
over one 16-bit stream.

Benchmark, RTX 4060, 4096x4096 layer, polar dual vs two FP16 cuBLAS GEMMs:

| M | polar dual | fp16 dual | ratio |
|---|---|---|---|
| 1 (decode) | 0.30 ms | 0.27 ms | 0.90x |
| 8 | 0.30 ms | 0.28 ms | 0.93x |
| 32 | 0.30 ms | 0.28 ms | 0.92x |
| 128 | 0.77 ms | 0.29 ms | 0.38x |
| 1024 (prefill) | 5.82 ms | 2.29 ms | 0.39x |

Reading the table honestly:

- At decode (M=1), **two models run at 90% of the cost of two FP16 models
  while reading half the weight traffic** (33.6 vs 67.1 MB per layer). Against
  a single FP16 model (~0.14 ms), the second model costs +0.16 ms.
- At prefill, cuBLAS wins. The matmul is compute-bound on tensor cores, and
  per-weight dequant ALU is pure overhead there. The format wins on traffic
  and memory, not on tensor cores - and the write-up does not pretend
  otherwise.

Optimization journey at M=1, included because the failures are instructive:

| Kernel version | M=1 ratio vs fp16 dual | Lesson |
|---|---|---|
| naive polar, cos/sin in the loop | 0.22x | transcendentals in the inner loop kill everything |
| LUT dequant + fixed per-group indexing | 0.22x | accuracy fixed (22% -> 2.4% rel. error), speed unchanged: wrong benchmark |
| dual matmul + INT16 phase | 0.35x | compare what the format is actually for: two models per pass |
| bit-packed uint16 + autotune | 0.83x | packing + tile tuning; LUTs live in L1 |
| **GEMV-optimized tiles (BLOCK_M=1)** | **0.90x** | **final: aggressive autotune with small M tiles** |

The remaining ~10% at decode is dequant ALU and gather latency; closing it
needs a hand-written CUDA kernel with warp-level fused dequant (see Future
work).

## 7. CPU: where the format honestly wins

On CPU (6 threads, PyTorch):

| | polar | fp32 dual |
|---|---|---|
| load traffic (2048x2048 layer) | 13 MB | 34 MB (2.56x less) |
| one-time dequant | 56 ms | - |
| per-token GEMV with per-call dequant | 45.6 ms | 1.6 ms |

The per-token row is the honest negative result: without a fused C++/AVX2
kernel, PyTorch materializes dequantized weights and loses 8-30x. The winning
CPU deployment pattern is therefore: dequantize once at startup, serve FP32 at
parity speed, and keep a 2.56x smaller RAM/disk footprint - which matters on
edge devices more than raw token rate.

## 8. Use cases

- Two specialists, one artifact: one download, one load path, router picks.
- Merging without identity loss: both personalities preserved exactly.
- Bandwidth-bound distribution: 6.6 GB instead of 11.8 GB.
- Free second opinion: abstention and escalation give a fallback answer at no
  extra storage cost.

## 9. Limitations

- Both models must share the same architecture (identical tensor layout).
- The GPU kernel is a research prototype: it wins on decode traffic, loses
  prefill to cuBLAS.
- The router is a calibrated heuristic, not a learned classifier.
- Quality is near-lossless, not lossless: expect < 0.6% PPL movement.

## 10. Future work

- Fused CUDA decode kernel (warp-level dequant) to close the remaining 10%.
- GGUF / llama.cpp port as a `Q_POLAR` block type, so the ecosystem can consume
  the format directly.
- Quantization-aware training in polar space (teach the model to live at
  6+10 bits).
- N-model packing: quaternions for 4 models per weight stream, at the cost of
  a tighter bit budget per model.

## Reproduction

```bash
pip install torch transformers numpy triton   # triton-windows on Windows
pytest test_polar_inference.py -v             # 36 passed

python polar_main.py pack --work-dir work --path1 <instruct> --path2 <coder>
python polar_main.py router --work-dir work --path1 <instruct> --path2 <coder> \
  --prompt "Write a Python function to reverse a linked list, then explain it."

python polar_kernels.py    # GPU benchmark table above
```

Hardware of record: RTX 4060 8 GB, 32 GB RAM, Python 3.12, torch 2.13,
triton 3.7 (Windows build).
```
