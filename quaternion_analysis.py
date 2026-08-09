#!/usr/bin/env python3
"""
Quaternion packing rate-distortion analysis - fast version.
Sample-based sweep + single full-data validation of best config.
"""
import numpy as np
import math
import time
from pathlib import Path

CHUNK = 100_000_000
SAMPLE_SIZE = 100_000_000
BLOCK = 500_000
MU = 255.0
VALIDATE_FULL = True


def memmap_model(work_dir, idx):
    return np.memmap(work_dir / f"model{idx}_weights.dat", dtype='float32', mode='r')


def collect_sample(w, seed):
    """Random contiguous blocks -> representative sample in RAM."""
    n = len(w)
    n_blocks = SAMPLE_SIZE // BLOCK
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - BLOCK, n_blocks)
    parts = [np.array(w[s:s + BLOCK], dtype=np.float32) for s in starts]
    return np.concatenate(parts)


def to_hyperspherical(w1, w2, w3, w4):
    r = np.sqrt(w1**2 + w2**2 + w3**2 + w4**2)
    t1 = np.arctan2(np.sqrt(w2**2 + w3**2 + w4**2), w1)
    t2 = np.arctan2(np.sqrt(w3**2 + w4**2), w2)
    t3 = np.mod(np.arctan2(w4, w3), 2 * np.pi)
    return r, t1, t2, t3


def from_hyperspherical(r, t1, t2, t3):
    return (r * np.cos(t1),
            r * np.sin(t1) * np.cos(t2),
            r * np.sin(t1) * np.sin(t2) * np.cos(t3),
            r * np.sin(t1) * np.sin(t2) * np.sin(t3))


def quantize_eval(w, r, t1, t2, t3, norm_bits, dir_bits, plug_cap):
    """Single-pass quantize + plug correction. w = list of 4 normalized arrays."""
    n = len(r)
    log_mu = math.log1p(MU)
    levels = 2 ** norm_bits - 1
    b1 = b2 = dir_bits // 3
    b3 = dir_bits - b1 - b2

    r_max = float(np.max(r)) * 1.001

    comp = np.log1p(MU * r / r_max) / log_mu
    r_rest = (np.exp(np.round(comp * levels) / levels * log_mu) - 1) * r_max / MU

    t1_rest = np.round(t1 / np.pi * (2**b1 - 1)) / (2**b1 - 1) * np.pi
    t2_rest = np.round(t2 / np.pi * (2**b2 - 1)) / (2**b2 - 1) * np.pi
    t3_rest = np.round(t3 / (2*np.pi) * (2**b3 - 1)) / (2**b3 - 1) * 2*np.pi

    rw = from_hyperspherical(r_rest, t1_rest, t2_rest, t3_rest)

    # Base accumulators (per model, float64)
    num = np.zeros(4); den_w = np.zeros(4); den_rw = np.zeros(4); sq_err = np.zeros(4)
    for k in range(4):
        num[k] = float(np.sum(w[k] * rw[k], dtype=np.float64))
        den_w[k] = float(np.sum(w[k] * w[k], dtype=np.float64))
        den_rw[k] = float(np.sum(rw[k] * rw[k], dtype=np.float64))
        sq_err[k] = float(np.sum((w[k] - rw[k])**2, dtype=np.float64))

    # Plug: worst tuples -> exact. Correction to accumulators (no 2nd pass)
    rel_err = np.sqrt(sum((w[k] - rw[k])**2 for k in range(4))) / np.maximum(r, 1e-12)
    thr = np.percentile(rel_err, (1 - plug_cap) * 100)
    idx = np.nonzero(rel_err > thr)[0]
    f = len(idx) / n
    for k in range(4):
        wk = w[k][idx]; rwk = rw[k][idx]
        num[k] += float(np.sum(wk*wk - wk*rwk))
        den_rw[k] += float(np.sum(wk*wk - rwk*rwk))
        sq_err[k] -= float(np.sum((wk - rwk)**2))

    cosines = [num[k] / math.sqrt(den_w[k] * den_rw[k]) for k in range(4)]
    rel_mse = [sq_err[k] / den_w[k] for k in range(4)]
    bits_per_weight = (norm_bits + dir_bits) / 4 + 16 * f

    return bits_per_weight, float(np.mean(cosines)), float(np.mean(rel_mse)), f


def main():
    t0 = time.time()
    print("=" * 70)
    print("QUATERNION R-D ANALYSIS (fast, sample-based)")
    print("=" * 70)

    models = [
        ('Qwen2.5-3B-Instruct',      memmap_model(Path('dual_pack_qwen_base'), 1)),
        ('Qwen2.5-Coder-3B-Instruct', memmap_model(Path('dual_pack_ministral_qwen'), 2)),
        ('Qwen2.5-3B (base)',         memmap_model(Path('dual_pack_qwen_base'), 2)),
        ('Ministral-3b-instruct',     memmap_model(Path('dual_pack_ministral_qwen'), 1)),
    ]
    n_min = min(len(m[1]) for m in models)

    print(f"\nSampling {SAMPLE_SIZE // 1_000_000}M weights per model...")
    w = [collect_sample(m[1], seed=42 + i) for i, m in enumerate(models)]

    # Per-model RMS normalization
    s = [math.sqrt(float(np.sum(wk*wk, dtype=np.float64)) / len(wk)) for wk in w]
    w = [wk / sk for wk, sk in zip(w, s)]
    print(f"  Per-model RMS scales: {[f'{x:.4f}' for x in s]}")

    # Distribution stats on sample
    r, t1, t2, t3 = to_hyperspherical(*w)
    sigma = math.sqrt(sum(float(np.var(wk)) for wk in w) / 4)
    dirs = np.stack(w, axis=1) / np.maximum(r, 1e-12)[:, None]
    print(f"\n  Norm/sigma: {float(np.mean(r))/sigma:.3f} (chi-4 ~1.88)")
    print(f"  Direction comp variance: {np.var(dirs, axis=0).round(4)} (uniform S3: 0.25)")
    thr_e = np.percentile(r, 99)
    print(f"  Top-1% norm energy: {np.sum(r[r>thr_e]**2)/np.sum(r**2)*100:.1f}%")

    # Sweep
    print("\n" + "=" * 70)
    print("BIT ALLOCATION SWEEP")
    print("=" * 70)
    print(f"{'Alloc':<8}{'Plug':<7}{'Bits/W':<8}{'Cosine':<10}{'RelMSE':<11}")
    print("-" * 50)

    results = []
    for norm_bits, dir_bits in [(6, 26), (7, 25), (8, 24)]:
        for plug_cap in [0.03, 0.05, 0.08]:
            bw, cos, mse, f = quantize_eval(w, r, t1, t2, t3, norm_bits, dir_bits, plug_cap)
            results.append((norm_bits, dir_bits, plug_cap, bw, cos, mse))
            print(f"{norm_bits}+{dir_bits:<4}{plug_cap:<7.2f}{bw:<8.2f}{cos:<10.6f}{mse:<11.2e}")

    print("\nDual-pack baseline (8 bits/model): cosine 0.9998, relMSE ~4e-4")
    print(f"Shannon bound R=8: relMSE 1.5e-5")
    print(f"\nSweep time: {time.time()-t0:.0f}s")

    if VALIDATE_FULL:
        best = min(results, key=lambda x: x[5])
        nb, db, pc = best[0], best[1], best[2]
        print("\n" + "=" * 70)
        print(f"FULL-DATA VALIDATION: {nb}+{db}, plug {pc}")
        print("=" * 70)

        # Two sub-passes: (1) global r_max + plug threshold via rel_err sample
        #     reuse sample: rerun quantize_eval to get thr
        log_mu = math.log1p(MU); levels = 2**nb - 1
        b1 = b2 = db // 3; b3 = db - b1 - b2

        r_max = float(np.max(r)) * 1.001
        comp = np.log1p(MU * r / r_max) / log_mu
        r_rest = (np.exp(np.round(comp*levels)/levels*log_mu) - 1) * r_max / MU
        rw_s = from_hyperspherical(
            r_rest,
            np.round(t1/np.pi*(2**b1-1))/(2**b1-1)*np.pi,
            np.round(t2/np.pi*(2**b2-1))/(2**b2-1)*np.pi,
            np.round(t3/(2*np.pi)*(2**b3-1))/(2**b3-1)*2*np.pi)
        rel_err_s = np.sqrt(sum((w[k]-rw_s[k])**2 for k in range(4)))/np.maximum(r,1e-12)
        thr = np.percentile(rel_err_s, (1 - pc) * 100)

        # RMS full
        s_full = []
        for name, m in models:
            acc = 0.0
            for i in range(0, n_min, CHUNK):
                c = np.asarray(m[i:min(i+CHUNK, n_min)], dtype=np.float64)
                acc += np.sum(c*c)
            s_full.append(math.sqrt(acc / n_min))

        num = np.zeros(4); den_w = np.zeros(4); den_rw = np.zeros(4); sq_err = np.zeros(4)
        n_plug = 0
        n_chunks = (n_min + CHUNK - 1) // CHUNK
        for ci, i in enumerate(range(0, n_min, CHUNK)):
            j = min(i + CHUNK, n_min)
            wc = [np.asarray(m[i:j], dtype=np.float32) / sk
                  for (name, m), sk in zip(models, s_full)]
            rc, t1c, t2c, t3c = to_hyperspherical(*wc)
            rr = (np.exp(np.round(np.log1p(MU*rc/r_max)/log_mu*levels)/levels*log_mu)-1)*r_max/MU
            rw = from_hyperspherical(
                rr,
                np.round(t1c/np.pi*(2**b1-1))/(2**b1-1)*np.pi,
                np.round(t2c/np.pi*(2**b2-1))/(2**b2-1)*np.pi,
                np.round(t3c/(2*np.pi)*(2**b3-1))/(2**b3-1)*2*np.pi)
            re_c = np.sqrt(sum((wc[k]-rw[k])**2 for k in range(4)))/np.maximum(rc,1e-12)
            mask = re_c > thr
            n_plug += int(np.sum(mask))
            for k in range(4):
                wk = wc[k]; rwk = rw[k].copy(); rwk[mask] = wk[mask]
                num[k] += float(np.sum(wk*rwk, dtype=np.float64))
                den_w[k] += float(np.sum(wk*wk, dtype=np.float64))
                den_rw[k] += float(np.sum(rwk*rwk, dtype=np.float64))
                sq_err[k] += float(np.sum((wk-rwk)**2, dtype=np.float64))
            if ci % 10 == 0:
                print(f"  Chunk {ci+1}/{n_chunks}")

        cosines = [num[k]/math.sqrt(den_w[k]*den_rw[k]) for k in range(4)]
        rel_mse = [sq_err[k]/den_w[k] for k in range(4)]
        f = n_plug / n_min
        print(f"\n  FULL RESULT: bits/W={(nb+db)/4 + 16*f:.2f}  "
              f"cosine={np.mean(cosines):.6f}  relMSE={np.mean(rel_mse):.2e}")

if __name__ == "__main__":
    main()