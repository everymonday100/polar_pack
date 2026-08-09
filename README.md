# polar_pack — pack multiple LLMs into one weight stream

**polar_pack** stores several same-family LLMs in a *single* quantized weight
stream by coding weight tuples in spherical coordinates:

- **dual-pack** (v1.0): 2 models → complex plane (2D polar), 16 bits/pair.
- **quad-pack** (v2.0): 4 models → 4D hypersphere (quaternion container),
  32 bits/tuple, byte-aligned uint32.

Fine-tunes of a common base are highly correlated, so weight tuples
concentrate near a low-dimensional manifold — spherical coding spends bits
on what actually varies. Result: near-lossless quality at ~8 bits/model, and
a decode kernel that reads **one** packed stream instead of N FP16 copies.

## Headline results (3B-class models, RTX 4060)

| Format | Models | Compression | Quality (PPL Δ) | Decode vs FP16 |
|--------|--------|-------------|------------------|----------------|
| dual   | 2 | 1.76–1.78× | 0.3–0.6% | 0.91× |
| quad   | 4 | 1.74× (24.7→14.4 GB) | **−0.002% (lossless)** | **1.6–2.0×** |

**Which to choose?** 2 models → dual. 4 models / best quality / fastest
decode → quad.

## How it works (30 seconds)

1. Corresponding weights `(w1..wN)` form a vector in R^N.
2. The vector is coded as norm + direction (polar in 2D, hyperspherical in 4D)
   with fixed-width codes → GPU random access preserved.
3. The heavy tail (outliers) goes to a **plug table** (exact FP16) — our
   random-access substitute for entropy coding.
4. Each model is restored by projecting back onto its own axis; the Triton
   kernel dequantizes on the fly, one memory pass → N matmuls.

Theory, rate–distortion analysis and full benchmarks: **WHITEPAPER.md**.

## Requirements

- Python 3.10+, `numpy`, `torch`, `transformers`
- `triton` + NVIDIA GPU for decode kernels (benchmarked on RTX 4060)

## Quick start — dual (2 models)

    python polar_main.py pack --work-dir work ^
      --path1 <modelA_safetensors_dir> --path2 <modelB_safetensors_dir>

    python polar_main.py unpack --work-dir work
    python measure_ppl.py        # quality check

Includes a calibrated router with abstention (single / parallel / refusal
escalation modes) — see v1.0 release notes.

## Quick start — quad (4 models)

Edit the model paths at the bottom of the scripts, then:

    python quaternion_packer.py   # pack 4 models -> polar.quadpack + unpack
    python measure_quad_ppl.py    # PPL: expect ~0% degradation
    python quad_kernels.py        # Triton bench: 1 pass -> 4 matmuls

## Compatibility

Models must produce the same flat Linear-weight stream (identical layer
shapes). Length mismatch is handled by per-model FP16 tails.
Tested: Qwen2.5-3B-Instruct, Qwen2.5-Coder-3B-Instruct, Qwen2.5-3B (base),
Ministral-3B-Instruct (cross-architecture pair included).

## Contents

| File | Purpose |
|------|---------|
| `polar_main.py` | dual CLI: pack / unpack / route |
| `polar_packer.py`, `polar_bitpack.py` | dual format core (chunked unpack) |
| `quaternion_packer.py` | quad format: .quadpack v2 pack/unpack |
| `quaternion_analysis.py` | rate–distortion analysis (sample-based) |
| `quad_kernels.py` | Triton decode kernel (4 matmuls / pass) |
| `measure_ppl.py`, `measure_quad_ppl.py` | quality measurement |
| `WHITEPAPER.md` | theory + full experimental tables |

## Releases

- **v2.0** — quad-pack: 4 models, statistical lossless, 1.6–2.0× faster decode.
- **v1.0** — dual-pack: 2 models, router with abstention.

## Roadmap

- [ ] GGUF / llama.cpp port (`Q_POLAR` / `Q_QUAD` block types)
- [ ] Group-reference spherical coding (aligned delta)
- [ ] Quantization-aware training in spherical space
- [ ] Fused CUDA decode kernel

## Citation

If you use polar_pack in research, cite the repository and WHITEPAPER.md
(sections 3–5).