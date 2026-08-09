#!/usr/bin/env python3
"""
.quadpack v1.1: 4-model quaternion packing, 8+24 byte-aligned + per-model tails.
uint32 word: [norm 8][t1 8][t2 8][t3 8]
File layout: header | codes | plug_idx | plug_vals | tail_0..3 (FP16)
"""
import numpy as np
import math
import struct
import time
from pathlib import Path

MAGIC = b'QUAD'
VERSION = 2
N_MODELS = 4
MU = 255.0
CHUNK = 100_000_000
HEADER_FMT = '<4sBBQBBBBfQ4ff4Q'
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def to_hyperspherical(w1, w2, w3, w4):
    r = np.sqrt(w1**2 + w2**2 + w3**2 + w4**2)
    t1 = np.arctan2(np.sqrt(w2**2 + w3**2 + w4**2), w1)
    t2 = np.arctan2(np.sqrt(w3**2 + w4**2), w2)
    t3 = np.mod(np.arctan2(w4, w3), 2 * np.pi)
    return r, t1, t2, t3


def quantize_chunk(wc, r_max, thr_collect=None):
    log_mu = math.log1p(MU)
    r, t1, t2, t3 = to_hyperspherical(*wc)

    nq = np.round(np.log1p(MU * r / r_max) / log_mu * 255).clip(0, 255).astype(np.uint32)
    q1 = np.round(t1 / np.pi * 255).clip(0, 255).astype(np.uint32)
    q2 = np.round(t2 / np.pi * 255).clip(0, 255).astype(np.uint32)
    q3 = np.round(t3 / (2 * np.pi) * 255).clip(0, 255).astype(np.uint32)

    codes = (nq << 24) | (q1 << 16) | (q2 << 8) | q3

    if thr_collect is not None:
        r_rest = (np.exp(nq / 255 * log_mu) - 1) * r_max / MU
        rw = (r_rest * np.cos(q1 / 255 * np.pi),
              r_rest * np.sin(q1 / 255 * np.pi) * np.cos(q2 / 255 * np.pi),
              r_rest * np.sin(q1 / 255 * np.pi) * np.sin(q2 / 255 * np.pi) * np.cos(q3 / 255 * 2 * np.pi),
              r_rest * np.sin(q1 / 255 * np.pi) * np.sin(q2 / 255 * np.pi) * np.sin(q3 / 255 * 2 * np.pi))
        rel_err = np.sqrt(sum((wc[k] - rw[k])**2 for k in range(4))) / np.maximum(r, 1e-12)
        return codes, rel_err
    return codes


def pack_quad(models, out_path, plug_cap=0.05):
    t0 = time.time()
    lengths = [len(m) for m in models]
    n = min(lengths)
    tails = [int(L - n) for L in lengths]
    out_path = Path(out_path)

    # Pass A: per-model RMS (on common part)
    print("Pass A: per-model RMS...")
    s = []
    for m in models:
        acc = 0.0
        for i in range(0, n, CHUNK):
            c = np.asarray(m[i:min(i + CHUNK, n)], dtype=np.float64)
            acc += np.sum(c * c)
        s.append(math.sqrt(acc / n))
    print(f"   RMS: {[f'{x:.4f}' for x in s]}, tails: {tails}")

    # Pass B: r_max + plug threshold
    print("Pass B: r_max + plug threshold...")
    r_max = 0.0
    for i in range(0, n, CHUNK):
        j = min(i + CHUNK, n)
        wc = [np.asarray(m[i:j], dtype=np.float32) / sk for m, sk in zip(models, s)]
        r_max = max(r_max, float(np.max(np.sqrt(sum(x**2 for x in wc)))))
    r_max *= 1.001

    rng = np.random.default_rng(42)
    starts = rng.integers(0, n - 1_000_000, 100)
    errs = []
    for st in starts:
        wc = [np.asarray(m[st:st + 1_000_000], dtype=np.float32) / sk
              for m, sk in zip(models, s)]
        _, e = quantize_chunk(wc, r_max, thr_collect=True)
        errs.append(e)
    thr = float(np.percentile(np.concatenate(errs), (1 - plug_cap) * 100))
    print(f"   r_max={r_max:.5f}, plug thr={thr:.5f}")

    # Pass C: pack codes + collect plug
    print("Pass C: packing...")
    with open(out_path, 'wb') as f:
        f.write(struct.pack(HEADER_FMT, MAGIC, VERSION, N_MODELS, n,
                            8, 8, 8, 8, MU, 0, *s, r_max, *tails))
        codes_mm = np.memmap(str(out_path), dtype=np.uint32, mode='r+',
                             offset=HEADER_SIZE, shape=(n,))
        plug_idx, plug_vals = [], []
        n_chunks = (n + CHUNK - 1) // CHUNK
        for ci, i in enumerate(range(0, n, CHUNK)):
            j = min(i + CHUNK, n)
            wc = [np.asarray(m[i:j], dtype=np.float32) / sk for m, sk in zip(models, s)]
            codes, rel_err = quantize_chunk(wc, r_max, thr_collect=True)
            codes_mm[i:j] = codes
            mask = rel_err > thr
            if mask.any():
                idx = np.nonzero(mask)[0] + i
                plug_idx.append(idx.astype(np.uint32))
                plug_vals.append(np.stack([wc[k][mask] for k in range(4)], axis=1)
                                 .astype(np.float16))
            if ci % 10 == 0:
                print(f"   Chunk {ci + 1}/{n_chunks}")
        codes_mm.flush()
        del codes_mm

        plug_idx = np.concatenate(plug_idx) if plug_idx else np.array([], np.uint32)
        plug_vals = np.concatenate(plug_vals) if len(plug_vals) else np.array([], np.float16).reshape(-1, 4)

        with open(out_path, 'ab') as f:
            f.write(plug_idx.tobytes())
            f.write(plug_vals.tobytes())
            # Tails (FP16) for models longer than n
            for k, m in enumerate(models):
                if tails[k]:
                    tail = np.asarray(m[n:], dtype=np.float16)
                    f.write(tail.tobytes())
                    print(f"   Tail model {k}: {tails[k]} weights")

    with open(out_path, 'r+b') as f:
        hdr = list(struct.unpack(HEADER_FMT, f.read(HEADER_SIZE)))
        hdr[9] = len(plug_idx)
        f.seek(0)
        f.write(struct.pack(HEADER_FMT, *hdr))

    total = out_path.stat().st_size
    total_weights = sum(lengths)
    bits_w = total * 8 / total_weights
    print(f"   [OK] {out_path.name}: {total / 1e9:.2f} GB, {bits_w:.2f} bits/weight, "
          f"plug {len(plug_idx) / n * 100:.1f}%, {time.time() - t0:.0f}s")


def unpack_quad(path, out_dir):
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(path, 'rb') as f:
        hdr = struct.unpack(HEADER_FMT, f.read(HEADER_SIZE))
    magic, ver, nm, n, nb, b1, b2, b3, mu, plug_n = hdr[:10]
    s = hdr[10:14]
    r_max = hdr[14]
    tails = hdr[15:19]
    assert magic == MAGIC and ver == VERSION
    log_mu = math.log1p(MU)

    codes_off = HEADER_SIZE
    plug_off = codes_off + n * 4
    tails_off = plug_off + plug_n * 4 + plug_n * 8

    codes = np.memmap(str(path), dtype=np.uint32, mode='r', offset=codes_off, shape=(n,))

    with open(path, 'rb') as f:
        f.seek(plug_off)
        plug_idx = np.frombuffer(f.read(plug_n * 4), dtype=np.uint32).copy()
        plug_vals = np.frombuffer(f.read(plug_n * 8), dtype=np.float16).reshape(-1, 4).copy()

    outs = [np.memmap(str(out_dir / f"qmodel{k}_restored.dat"), dtype='float32',
                      mode='w+', shape=(n + tails[k],)) for k in range(4)]

    n_chunks = (n + CHUNK - 1) // CHUNK
    for ci, i in enumerate(range(0, n, CHUNK)):
        j = min(i + CHUNK, n)
        c = codes[i:j]
        nq = c >> 24
        q1 = (c >> 16) & 255
        q2 = (c >> 8) & 255
        q3 = c & 255

        r_rest = (np.exp(nq / 255 * log_mu) - 1) * r_max / MU
        t1 = q1 / 255 * np.pi
        t2 = q2 / 255 * np.pi
        t3 = q3 / 255 * 2 * np.pi
        rw = (r_rest * np.cos(t1),
              r_rest * np.sin(t1) * np.cos(t2),
              r_rest * np.sin(t1) * np.sin(t2) * np.cos(t3),
              r_rest * np.sin(t1) * np.sin(t2) * np.sin(t3))

        if plug_n:
            lo = np.searchsorted(plug_idx, i)
            hi = np.searchsorted(plug_idx, j)
            if hi > lo:
                pos = plug_idx[lo:hi].astype(np.int64) - i
                for k in range(4):
                    rw[k][pos] = plug_vals[lo:hi, k].astype(np.float32)

        for k in range(4):
            outs[k][i:j] = rw[k] * s[k]

        if ci % 10 == 0:
            print(f"   Chunk {ci + 1}/{n_chunks}")

    # Write tails
    with open(path, 'rb') as f:
        off = tails_off
        for k in range(4):
            if tails[k]:
                f.seek(off)
                tail = np.frombuffer(f.read(tails[k] * 2), dtype=np.float16).astype(np.float32)
                outs[k][n:n + tails[k]] = tail
                off += tails[k] * 2
                print(f"   Tail model {k}: {tails[k]} weights restored")

    for o in outs:
        o.flush()
    print(f"   [OK] Unpacked 4 models to {out_dir}")


if __name__ == "__main__":
    models = [
        np.memmap('dual_pack_qwen_base/model1_weights.dat', dtype='float32', mode='r'),
        np.memmap('dual_pack_ministral_qwen/model2_weights.dat', dtype='float32', mode='r'),
        np.memmap('dual_pack_qwen_base/model2_weights.dat', dtype='float32', mode='r'),
        np.memmap('dual_pack_ministral_qwen/model1_weights.dat', dtype='float32', mode='r'),
    ]

    pack_quad(models, 'polar.quadpack', plug_cap=0.05)
    unpack_quad('polar.quadpack', 'quad_restored')