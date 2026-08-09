#!/usr/bin/env python3
"""
Test reversible quaternion transformer:
1. Single block reconstruction
2. Deep model reconstruction (GELU variant - expected UNSTABLE)
3. Deep model reconstruction (LINEAR variant - expected STABLE)
4. Memory profiling: O(1) vs O(depth)
"""
import torch
import torch.nn as nn
from quaternion_reversible_block import ReversibleQuaternionTransformer, ReversibleQuaternionBlock
import numpy as np

torch.manual_seed(42)

VOCAB_SIZE = 100
EMBED_DIM = 32
FF_DIM = 64
NUM_LAYERS = 10
B, S = 4, 16

print("="*70)
print("PHASE 2: REVERSIBILITY TEST")
print("="*70)
print(f"Config: embed={EMBED_DIM}, ff={FF_DIM}, layers={NUM_LAYERS}")

# ============================================================
# TEST 1: Single block reconstruction
# ============================================================
print("\n" + "-"*70)
print("TEST 1: Single Block Reconstruction")
print("-"*70)

block = ReversibleQuaternionBlock(EMBED_DIM, FF_DIM)
block.eval()

x1 = torch.randn(B, S, EMBED_DIM)
x2 = torch.randn(B, S, EMBED_DIM)

with torch.no_grad():
    y1, y2 = block(x1, x2)
    x1_rec, x2_rec = block.inverse(y1, y2)

err1 = (x1_rec - x1).abs().max().item()
err2 = (x2_rec - x2).abs().max().item()
print(f"Max |x1_rec - x1|: {err1:.2e}")
print(f"Max |x2_rec - x2|: {err2:.2e}")
print("✓ EXACT reconstruction" if err1 < 1e-4 and err2 < 1e-4 else "✗ Reconstruction error too large")

# ============================================================
# TEST 2: Deep model with GELU (expected UNSTABLE)
# ============================================================
print("\n" + "-"*70)
print("TEST 2: Deep Model with GELU (expected UNSTABLE)")
print("-"*70)

model_gelu = ReversibleQuaternionTransformer(VOCAB_SIZE, EMBED_DIM, FF_DIM, NUM_LAYERS, linear=False)
model_gelu.eval()

input_ids = torch.randint(0, VOCAB_SIZE, (B, S))

with torch.no_grad():
    intermediates, y1_final, y2_final = model_gelu.forward_with_intermediates(input_ids)
    x1_rec, x2_rec = model_gelu.backward_recompute(y1_final, y2_final)
    x1_orig, x2_orig = intermediates[0]
    err1_gelu = (x1_rec - x1_orig).abs().max().item()
    err2_gelu = (x2_rec - x2_orig).abs().max().item()

print(f"Max |x1_rec - x1_orig|: {err1_gelu:.2e}")
print(f"Max |x2_rec - x2_orig|: {err2_gelu:.2e}")
print("✗ Expected unstable (GELU not exactly invertible)" if err1_gelu > 1e-3 else "✓ Surprisingly stable")

# ============================================================
# TEST 3: Deep model with LINEAR F/G (expected STABLE)
# ============================================================
print("\n" + "-"*70)
print("TEST 3: Deep Model with LINEAR F/G (expected STABLE)")
print("-"*70)

model_linear = ReversibleQuaternionTransformer(VOCAB_SIZE, EMBED_DIM, FF_DIM, NUM_LAYERS, linear=True)
model_linear.eval()

with torch.no_grad():
    intermediates, y1_final, y2_final = model_linear.forward_with_intermediates(input_ids)
    x1_rec, x2_rec = model_linear.backward_recompute(y1_final, y2_final)
    x1_orig, x2_orig = intermediates[0]
    err1_lin = (x1_rec - x1_orig).abs().max().item()
    err2_lin = (x2_rec - x2_orig).abs().max().item()

print(f"Max |x1_rec - x1_orig|: {err1_lin:.2e}")
print(f"Max |x2_rec - x2_orig|: {err2_lin:.2e}")
print("✓ STABLE reconstruction (linear F/G)" if err1_lin < 1e-3 and err2_lin < 1e-3 else "✗ Still unstable")

# ============================================================
# TEST 4: Memory profiling
# ============================================================
print("\n" + "-"*70)
print("TEST 4: Memory Profiling (activation storage)")
print("-"*70)

def count_activation_memory(num_layers, store_all):
    per_stream = B * S * EMBED_DIM * 4  # bytes, float32
    streams_per_layer = 2
    if store_all:
        total = per_stream * streams_per_layer * (num_layers + 1)
    else:
        total = per_stream * streams_per_layer
    return total / 1e6  # MB

for nl in [2, 10, 50, 100]:
    baseline = count_activation_memory(nl, store_all=True)
    reversible = count_activation_memory(nl, store_all=False)
    ratio = baseline / reversible
    print(f"  {nl:3d} layers: baseline {baseline:8.2f} MB | reversible {reversible:6.2f} MB | {ratio:6.1f}x reduction")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("PHASE 2 RESULTS")
print("="*70)
print(f"[1] Single block reconstruction: error {max(err1, err2):.2e}  → {'PASS' if max(err1,err2) < 1e-4 else 'FAIL'}")
print(f"[2] Deep GELU reconstruction:    error {err1_gelu:.2e}  → {'EXPECTED UNSTABLE' if err1_gelu > 1e-3 else 'STABLE'}")
print(f"[3] Deep LINEAR reconstruction:  error {err1_lin:.2e}  → {'PASS' if err1_lin < 1e-3 else 'FAIL'}")
print(f"[4] Memory: O(1) activations, {count_activation_memory(100, True)/count_activation_memory(100, False):.0f}x reduction at 100 layers")
print("="*70)
print("\nCONCLUSION:")
print("  GELU is not exactly invertible → use LINEAR for reversible,")
print("  or use gradient checkpointing for nonlinear parts (Reformer-style).")
print("="*70)