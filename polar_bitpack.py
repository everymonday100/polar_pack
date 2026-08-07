#!/usr/bin/env python3
"""
================================================================
POLAR BITPACK - actual .dualpack format
================================================================
Lossless serialization of lossy codes:
  weight pair (w1, w2) -> 2 bytes: [amp 6 bits | phase 10 bits]
  + fp16 group scales (1 per 32 pairs)
  + plug table: (uint32 index, fp16 exact amplitude)

Layout:
  [header struct][codes 2B*n_pairs][scales 2B*n_groups]
  [plug_idx 4B*plug_n][plug_val 2B*plug_n]

Result: ~2.24 bytes/pair = 8.97 bits/weight, compression ~1.78x
(two FP16 models in ~1.12 places of one)
================================================================
"""
import struct
import math
import json
import time
import numpy as np
from pathlib import Path

MAGIC = b'DUALPACK'
VERSION = 2
HEADER_FMT = '<8s I Q Q Q B B f I Q Q'
# magic, version, n_pairs, n1, n2, amp_bits, phase_bits, mu, group_size, chunk_size, plug_n

# ============================================================
# CODEC WITH CODE RETURN (amplitude)
# ============================================================
def quantize_amplitude_codes(amp_flat: np.ndarray, config):
    """
    Like quantize_amplitude_production, but returns codes for bit-packing.
    Returns: (amp_q uint8 FLAT, bmax_fp16, plug_idx uint32, plug_val fp16, decoded)
    """
    levels = 2 ** config.amp_bits - 1
    ol = amp_flat.size
    gs = config.group_size

    pad = (gs - ol % gs) % gs
    if pad:
        amp_flat = np.pad(amp_flat, (0, pad))
    b = amp_flat.reshape(-1, gs)

    bmax = np.abs(b).max(axis=1, keepdims=True)
    bmax = np.where(bmax > 1e-10, bmax, 1.0)
    norm = b / bmax

    log_mu = np.log1p(config.mu)
    compressed = np.sign(norm) * np.log1p(config.mu * np.abs(norm)) / log_mu

    # codes in 2D (groups x gs) - needed for per-group decoded decompression
    amp_q2d = np.round(compressed * levels).clip(0, levels).astype(np.uint8)

    restored_norm = np.expm1(amp_q2d.astype(np.float64) / levels * log_mu) / config.mu
    decoded = (restored_norm * bmax).ravel()[:ol].copy()

    # IMPORTANT: codes go out FLAT, length ol
    amp_q = amp_q2d.ravel()[:ol]

    plug_idx = np.array([], dtype=np.uint32)
    plug_val = np.array([], dtype=np.float16)
    if config.plug_cap > 0:
        flat_core = b.ravel()[:ol]
        e = (flat_core - decoded) ** 2
        n_plug = int(config.plug_cap * ol)
        if n_plug > 0:
            sel = np.argpartition(e, -n_plug)[-n_plug:]
            decoded[sel] = flat_core[sel]
            plug_idx = sel.astype(np.uint32)
            plug_val = flat_core[sel].astype(np.float16)

    return amp_q, bmax.astype(np.float16).ravel(), plug_idx, plug_val, decoded

# ============================================================
# BYTE SERIALIZATION (explicit big-endian per pair, portable)
# ============================================================
def _interleave(combined_u32: np.ndarray) -> np.ndarray:
    out = np.empty(combined_u32.size * 2, dtype=np.uint8)
    out[0::2] = ((combined_u32 >> 8) & 0xFF).astype(np.uint8)
    out[1::2] = (combined_u32 & 0xFF).astype(np.uint8)
    return out

def _deinterleave(buf_u8: np.ndarray) -> np.ndarray:
    return (buf_u8[0::2].astype(np.uint32) << 8) | buf_u8[1::2].astype(np.uint32)

# ============================================================
# PACK INTO .dualpack
# ============================================================
def pack_dual_models_packed(file1, file2, n1, n2, out_packed: Path, config):
    from polar_packer import PackingStats

    phase_bits = 16 - config.amp_bits
    assert 2 ** phase_bits >= config.num_phases, "amp_bits + phase_bits must be 16"

    fp1 = np.memmap(str(file1), dtype='float32', mode='r')
    fp2 = np.memmap(str(file2), dtype='float32', mode='r')

    max_n = max(n1, n2)
    n_chunks = (max_n + config.chunk_size - 1) // config.chunk_size

    codes_parts, scales_parts, plug_i_parts, plug_v_parts = [], [], [], []
    sum_dot1 = sum_sq1_o = sum_sq1_r = 0.0
    sum_dot2 = sum_sq2_o = sum_sq2_r = 0.0
    t0 = time.time()

    for chunk_idx in range(n_chunks):
        start = chunk_idx * config.chunk_size
        end = min(start + config.chunk_size, max_n)
        w1 = fp1[start:min(end, n1)].copy() if start < n1 else np.array([], np.float32)
        w2 = fp2[start:min(end, n2)].copy() if start < n2 else np.array([], np.float32)
        L = max(len(w1), len(w2))
        if len(w1) < L: w1 = np.pad(w1, (0, L - len(w1)))
        if len(w2) < L: w2 = np.pad(w2, (0, L - len(w2)))

        # Polar codes
        amplitude = np.sqrt(w1 ** 2 + w2 ** 2)
        phase = np.arctan2(w2, w1)
        phase_idx = (np.round((phase + np.pi) / (2 * np.pi) * config.num_phases)
                     .astype(np.int64) % config.num_phases).astype(np.uint32)

        amp_q, bmax16, plug_idx, plug_val, decoded = quantize_amplitude_codes(amplitude, config)
        if plug_idx.size:
            plug_i_parts.append(plug_idx + np.uint32(start))
            plug_v_parts.append(plug_val)

        combined = (amp_q.astype(np.uint32) << phase_bits) | phase_idx
        codes_parts.append(_interleave(combined))
        scales_parts.append(bmax16)

        # Metrics (reconstruction as in unpack)
        amp = decoded
        if plug_idx.size:
            amp = amp.copy()
            amp[plug_idx] = plug_val.astype(np.float32)
        ang = phase_idx.astype(np.float64) / config.num_phases * 2 * np.pi - np.pi
        w1r = (amp * np.cos(ang)).astype(np.float32)
        w2r = (amp * np.sin(ang)).astype(np.float32)
        n1c = min(end, n1) - start if start < n1 else 0
        n2c = min(end, n2) - start if start < n2 else 0
        if n1c > 0:
            o, r = w1[:n1c], w1r[:n1c]
            sum_dot1 += float((o * r).sum()); sum_sq1_o += float((o * o).sum()); sum_sq1_r += float((r * r).sum())
        if n2c > 0:
            o, r = w2[:n2c], w2r[:n2c]
            sum_dot2 += float((o * r).sum()); sum_sq2_o += float((o * o).sum()); sum_sq2_r += float((r * r).sum())

        if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
            print(f"   Chunk {chunk_idx + 1}/{n_chunks}")

    plug_i = np.concatenate(plug_i_parts) if plug_i_parts else np.array([], np.uint32)
    plug_v = np.concatenate(plug_v_parts) if plug_v_parts else np.array([], np.float16)
    order = np.argsort(plug_i)
    plug_i, plug_v = plug_i[order], plug_v[order]

    codes = np.concatenate(codes_parts)
    scales = np.concatenate(scales_parts)

    header = struct.pack(
        HEADER_FMT, MAGIC, VERSION, max_n, n1, n2,
        config.amp_bits, phase_bits, float(config.mu),
        config.group_size, config.chunk_size, plug_i.size
    )
    with open(out_packed, 'wb') as f:
        f.write(header)
        codes.tofile(f)
        scales.tofile(f)
        plug_i.tofile(f)
        plug_v.tofile(f)

    packed_bytes = len(header) + codes.nbytes + scales.nbytes + plug_i.nbytes + plug_v.nbytes
    original_bytes = (n1 + n2) * 2
    cos1 = sum_dot1 / (math.sqrt(sum_sq1_o) * math.sqrt(sum_sq1_r) + 1e-10)
    cos2 = sum_dot2 / (math.sqrt(sum_sq2_o) * math.sqrt(sum_sq2_r) + 1e-10)

    return PackingStats(
        n1=n1, n2=n2, amp_bits=config.amp_bits, num_phases=config.num_phases,
        mu=config.mu, plug_cap=config.plug_cap,
        packed_size_mb=packed_bytes / 1024 ** 2,
        original_size_mb=original_bytes / 1024 ** 2,
        compression_ratio=original_bytes / packed_bytes,
        bits_per_weight=packed_bytes * 8 / (n1 + n2),
        mse1=0.0, mse2=0.0, cos1=cos1, cos2=cos2,
        pack_time_seconds=round(time.time() - t0, 2),
        throughput_mweights_per_sec=round(n1 / (time.time() - t0) / 1e6, 2)
    )

# ============================================================
# UNPACK FROM .dualpack
# ============================================================
def unpack_dual_models(packed: Path, out1: Path, out2: Path):
    with open(packed, 'rb') as f:
        hs = struct.calcsize(HEADER_FMT)
        (magic, ver, n_pairs, n1, n2, amp_bits, phase_bits, mu,
         gs, chunk_size, plug_n) = struct.unpack(HEADER_FMT, f.read(hs))
        assert magic == MAGIC and ver == VERSION, "wrong .dualpack format"

        codes = _deinterleave(np.frombuffer(f.read(n_pairs * 2), dtype=np.uint8))
        n_groups = 0
        for s in range(0, n_pairs, chunk_size):
            n_groups += math.ceil(min(chunk_size, n_pairs - s) / gs)
        scales = np.frombuffer(f.read(n_groups * 2), dtype=np.float16).astype(np.float32)
        plug_i = np.frombuffer(f.read(plug_n * 4), dtype=np.uint32).copy()
        plug_v = np.frombuffer(f.read(plug_n * 2), dtype=np.float16).copy()

    levels = 2 ** amp_bits - 1
    log_mu = np.log1p(mu)
    phase_mask = (1 << phase_bits) - 1
    amp_q = (codes >> phase_bits) & levels
    phase_idx = codes & phase_mask

    out1_fp = np.memmap(str(out1), dtype='float32', mode='w+', shape=(n1,))
    out2_fp = np.memmap(str(out2), dtype='float32', mode='w+', shape=(n2,))

    g_off = 0
    p_lo = np.searchsorted(plug_i, 0)
    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        L = end - start

        amp_norm = np.expm1(amp_q[start:end].astype(np.float64) / levels * log_mu) / mu
        n_g = math.ceil(L / gs)
        scale = np.repeat(scales[g_off:g_off + n_g], gs)[:L]
        g_off += n_g
        amp = (amp_norm * scale).astype(np.float32)

        p_hi = np.searchsorted(plug_i, end)
        if p_hi > p_lo:
            amp[plug_i[p_lo:p_hi] - start] = plug_v[p_lo:p_hi].astype(np.float32)
        p_lo = p_hi

        ang = phase_idx[start:end].astype(np.float64) / (phase_mask + 1) * 2 * np.pi - np.pi
        w1 = amp * np.cos(ang)
        w2 = amp * np.sin(ang)

        if start < n1:
            out1_fp[start:min(end, n1)] = w1[:min(end, n1) - start]
        if start < n2:
            out2_fp[start:min(end, n2)] = w2[:min(end, n2) - start]

    out1_fp.flush()
    out2_fp.flush()
    del out1_fp, out2_fp
    print(f"   [OK] Unpacked: {out1.name}, {out2.name}")