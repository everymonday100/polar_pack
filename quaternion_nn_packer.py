#!/usr/bin/env python3
"""
.quatpack: quaternion NN weight packing (algebraic compression).
Each quaternion weight (r,i,j,k) -> spherical (norm, t1, t2, t3) -> uint32.

Two modes:
  - full: norm 8 + t1 8 + t2 8 + t3 8  (arbitrary quaternions)
  - unit: t1 11 + t2 10 + t3 11        (unit quaternions |q|=1, norm implicit)

Compression: 16 bytes (4 floats) -> 4 bytes = 4x
"""
import numpy as np
import math
import struct
import time
from pathlib import Path

MAGIC = b'QUAT'
VERSION = 1
MU = 255.0
HEADER_FMT = '<4sBBQBBfQf'  # magic, ver, mode, n, b1, b2, mu, plug_n, r_max
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def to_hyperspherical(w1, w2, w3, w4):
    r = np.sqrt(w1**2 + w2**2 + w3**2 + w4**2)
    t1 = np.arctan2(np.sqrt(w2**2 + w3**2 + w4**2), w1)
    t2 = np.arctan2(np.sqrt(w3**2 + w4**2), w2)
    t3 = np.mod(np.arctan2(w4, w3), 2 * np.pi)
    return r, t1, t2, t3


def pack_quat(quaternions, out_path, mode='full', plug_cap=0.05):
    """
    Pack quaternion weights (N, 4) into .quatpack.
    quaternions: ndarray (N, 4), each row = (r, i, j, k)
    mode: 'full' (norm+angles) or 'unit' (angles only, |q|=1)
    """
    t0 = time.time()
    quaternions = np.asarray(quaternions, dtype=np.float32)
    n = len(quaternions)
    w1, w2, w3, w4 = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    out_path = Path(out_path)

    r, t1, t2, t3 = to_hyperspherical(w1, w2, w3, w4)

    if mode == 'unit':
        # Normalize to unit sphere
        r_safe = np.maximum(r, 1e-12)
        w1, w2, w3, w4 = w1/r_safe, w2/r_safe, w3/r_safe, w4/r_safe
        r = np.ones_like(r)
        _, t1, t2, t3 = to_hyperspherical(w1, w2, w3, w4)
        # Higher resolution: t1 11, t2 10, t3 11
        q1 = np.round(t1 / np.pi * 2047).clip(0, 2047).astype(np.uint32)
        q2 = np.round(t2 / np.pi * 1023).clip(0, 1023).astype(np.uint32)
        q3 = np.round(t3 / (2*np.pi) * 2047).clip(0, 2047).astype(np.uint32)
        codes = (q1 << 21) | (q2 << 11) | q3
        b1, b2 = 11, 10
        r_max = 1.0
    else:
        # full mode: norm 8 + angles 8+8+8
        r_max = float(r.max()) * 1.001
        log_mu = math.log1p(MU)
        nq = np.round(np.log1p(MU * r / r_max) / log_mu * 255).clip(0, 255).astype(np.uint32)
        q1 = np.round(t1 / np.pi * 255).clip(0, 255).astype(np.uint32)
        q2 = np.round(t2 / np.pi * 255).clip(0, 255).astype(np.uint32)
        q3 = np.round(t3 / (2*np.pi) * 255).clip(0, 255).astype(np.uint32)
        codes = (nq << 24) | (q1 << 16) | (q2 << 8) | q3
        b1, b2 = 8, 8

    # Write file (no plug table for prototype)
    with open(out_path, 'wb') as f:
        f.write(struct.pack(HEADER_FMT, MAGIC, VERSION, 0 if mode=='full' else 1,
                            n, b1, b2, MU, 0, r_max))
        f.write(codes.astype(np.uint32).tobytes())

    total = out_path.stat().st_size
    raw = n * 4 * 4  # 4 floats x 4 bytes
    print(f"[OK] {out_path.name}: mode={mode}, n={n}, {total/1e6:.2f} MB, "
          f"compression {raw/total:.2f}x, {time.time()-t0:.1f}s")
    return out_path


def unpack_quat(path):
    """Unpack .quatpack -> quaternion weights (N, 4)."""
    path = Path(path)
    with open(path, 'rb') as f:
        hdr = struct.unpack(HEADER_FMT, f.read(HEADER_SIZE))
    magic, ver, mode, n, b1, b2, mu, plug_n, r_max = hdr
    assert magic == MAGIC

    codes = np.frombuffer(path.read_bytes()[HEADER_SIZE:], dtype=np.uint32, count=n)

    if mode == 1:  # unit
        q1 = (codes >> 21) & 2047
        q2 = (codes >> 11) & 1023
        q3 = codes & 2047
        t1 = q1 / 2047 * np.pi
        t2 = q2 / 1023 * np.pi
        t3 = q3 / 2047 * 2 * np.pi
        r = np.ones(n, dtype=np.float32)
    else:  # full
        log_mu = math.log1p(MU)
        nq = codes >> 24
        q1 = (codes >> 16) & 255
        q2 = (codes >> 8) & 255
        q3 = codes & 255
        r = (np.exp(nq / 255 * log_mu) - 1) * r_max / MU
        t1 = q1 / 255 * np.pi
        t2 = q2 / 255 * np.pi
        t3 = q3 / 255 * 2 * np.pi

    w1 = r * np.cos(t1)
    w2 = r * np.sin(t1) * np.cos(t2)
    w3 = r * np.sin(t1) * np.sin(t2) * np.cos(t3)
    w4 = r * np.sin(t1) * np.sin(t2) * np.sin(t3)

    return np.stack([w1, w2, w3, w4], axis=1).astype(np.float32)


if __name__ == "__main__":
    print("="*70)
    print(".QUATPACK SMOKE TEST")
    print("="*70)

    rng = np.random.default_rng(42)

    # Test 1: random quaternions (full mode)
    print("\n[Test 1] Random quaternions, full mode")
    quats = rng.normal(0, 0.02, (100000, 4)).astype(np.float32)
    pack_quat(quats, 'test_full.quatpack', mode='full')
    rec = unpack_quat('test_full.quatpack')
    err = np.linalg.norm(rec - quats) / np.linalg.norm(quats)
    cos = np.sum(rec*quats) / (np.linalg.norm(rec)*np.linalg.norm(quats))
    print(f"  RelMSE: {err**2:.2e}, cosine: {cos:.6f}")

    # Test 2: unit quaternions (unit mode)
    print("\n[Test 2] Unit quaternions, unit mode")
    quats_u = rng.normal(0, 1, (100000, 4)).astype(np.float32)
    quats_u /= np.linalg.norm(quats_u, axis=1, keepdims=True)
    pack_quat(quats_u, 'test_unit.quatpack', mode='unit')
    rec_u = unpack_quat('test_unit.quatpack')
    err_u = np.linalg.norm(rec_u - quats_u) / np.linalg.norm(quats_u)
    cos_u = np.sum(rec_u*quats_u) / (np.linalg.norm(rec_u)*np.linalg.norm(quats_u))
    print(f"  RelMSE: {err_u**2:.2e}, cosine: {cos_u:.6f}")

    # Test 3: real weights from Orkis QuaternionLinear
    print("\n[Test 3] Real QuaternionLinear weights")
    import sys
    sys.path.insert(0, r'E:\OllamaModels\Pytorch-Quaternion-Neural-Networks')
    import torch
    from core_qnn.quaternion_layers import QuaternionLinear
    layer = QuaternionLinear(32, 16)
    # Stack 4 components into (N, 4)
    rw = layer.r_weight.data.flatten()
    iw = layer.i_weight.data.flatten()
    jw = layer.j_weight.data.flatten()
    kw = layer.k_weight.data.flatten()
    real_quats = torch.stack([rw, iw, jw, kw], dim=1).numpy()
    pack_quat(real_quats, 'test_real.quatpack', mode='full')
    rec_r = unpack_quat('test_real.quatpack')
    err_r = np.linalg.norm(rec_r - real_quats) / np.linalg.norm(real_quats)
    print(f"  RelMSE: {err_r**2:.2e}")

    print("\n" + "="*70)
    print("QUATPACK READY")
    print("="*70)