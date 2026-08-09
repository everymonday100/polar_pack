#!/usr/bin/env python3
"""
Phase 3: Integration with debugging.
"""
import torch
import torch.nn as nn
import numpy as np
from quaternion_reversible_block import ReversibleQuaternionTransformer
from quaternion_nn_packer import pack_quat, unpack_quat
from quaternion_linear_unit import QuaternionLinearUnit

torch.manual_seed(42)

VOCAB_SIZE = 100
EMBED_DIM = 32
FF_DIM = 64
NUM_LAYERS = 4
B, S = 4, 32

print("="*70)
print("PHASE 3: INTEGRATION (with debugging)")
print("="*70)

# STEP 1: Train
print("\n[Step 1] Training...")
model = ReversibleQuaternionTransformer(VOCAB_SIZE, EMBED_DIM, FF_DIM, NUM_LAYERS, linear=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

X = torch.randint(0, VOCAB_SIZE, (B, S))
Y = torch.randint(0, VOCAB_SIZE, (B, S))

for step in range(50):
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits.view(-1, VOCAB_SIZE), Y.view(-1))
    loss.backward()
    optimizer.step()

print(f"  Final loss: {loss.item():.4f}")

# STEP 2: Extract and pack (с отладкой)
print("\n[Step 2] Packing...")

all_quats = []
layer_info = []

for name, module in model.named_modules():
    if isinstance(module, QuaternionLinearUnit):
        q = module.packed_quaternions()
        all_quats.append(q)
        layer_info.append({
            'name': name,
            'in_q': module.in_q,
            'out_q': module.out_q,
            'n_quats': len(q)
        })
        print(f"  {name}: {module.in_q}x{module.out_q} = {len(q)} quaternions")

all_quats_array = np.concatenate(all_quats, axis=0)
pack_quat(all_quats_array, 'phase3_model.quatpack', mode='unit')

# STEP 3: Load (исправленный)
print("\n[Step 3] Loading...")

restored_quats = unpack_quat('phase3_model.quatpack')

model_restored = ReversibleQuaternionTransformer(VOCAB_SIZE, EMBED_DIM, FF_DIM, NUM_LAYERS, linear=True)

offset = 0
quat_idx = 0  # ← отдельный счётчик для QuaternionLinearUnit

for name, module in model_restored.named_modules():
    if isinstance(module, QuaternionLinearUnit):
        info = layer_info[quat_idx]  # ← используем quat_idx, не enumerate idx
        assert name == info['name'], f"Layer order mismatch: {name} vs {info['name']}"
        
        n_quats = info['n_quats']
        q = restored_quats[offset:offset + n_quats]
        
        q_reshaped = q.reshape(info['in_q'], info['out_q'], 4)
        
        with torch.no_grad():
            module.r_weight.copy_(torch.from_numpy(q_reshaped[..., 0]))
            module.i_weight.copy_(torch.from_numpy(q_reshaped[..., 1]))
            module.j_weight.copy_(torch.from_numpy(q_reshaped[..., 2]))
            module.k_weight.copy_(torch.from_numpy(q_reshaped[..., 3]))
        
        if quat_idx < 2:
            orig_q = all_quats_array[offset:offset + n_quats]
            diff = np.abs(orig_q - q).max()
            print(f"  {name}: max |orig - loaded| = {diff:.2e}")
        
        offset += n_quats
        quat_idx += 1  # ← инкрементируем только для QuaternionLinearUnit

print(f"  Loaded {quat_idx} layers")

# Скопировать НЕ-quaternion слои напрямую
print("\n[Step 3b] Copying non-quaternion layers (embed, out_proj)...")
with torch.no_grad():
    model_restored.embed.weight.copy_(model.embed.weight)
    model_restored.out_proj.weight.copy_(model.out_proj.weight)
    model_restored.out_proj.bias.copy_(model.out_proj.bias)
print("  Copied embed + out_proj")

# STEP 4: Compare (с отладкой)
print("\n[Step 4] Comparing outputs...")

model.eval()
model_restored.eval()

# DEBUG: compare first layer output
with torch.no_grad():
    x_test = torch.randn(1, 10, EMBED_DIM)
    
    # Get first quaternion linear layer from each model
    orig_first_layer = None
    rest_first_layer = None
    for module in model.modules():
        if isinstance(module, QuaternionLinearUnit):
            orig_first_layer = module
            break
    for module in model_restored.modules():
        if isinstance(module, QuaternionLinearUnit):
            rest_first_layer = module
            break
    
    out_orig = orig_first_layer(x_test)
    out_rest = rest_first_layer(x_test)
    
    first_layer_diff = (out_orig - out_rest).abs().max().item()
    print(f"  First layer output diff: {first_layer_diff:.2e}")

with torch.no_grad():
    logits_orig = model(X)
    logits_restored = model_restored(X)

diff = (logits_orig - logits_restored).abs().max().item()
rel_err = diff / (logits_orig.abs().max().item() + 1e-8)

print(f"  Max |logits_orig - logits_restored|: {diff:.2e}")
print(f"  Relative error: {rel_err:.2e}")

if rel_err < 0.01:
    print("  ✓ PASS")
else:
    print(f"  ✗ FAIL: {rel_err:.2%}")

# STEP 5: Reversible
print("\n[Step 5] Reversible inference...")

with torch.no_grad():
    intermediates, y1_final, y2_final = model_restored.forward_with_intermediates(X)
    x1_rec, x2_rec = model_restored.backward_recompute(y1_final, y2_final)
    x1_orig, x2_orig = intermediates[0]
    err1 = (x1_rec - x1_orig).abs().max().item()
    err2 = (x2_rec - x2_orig).abs().max().item()

print(f"  Reversible error: {max(err1, err2):.2e}")
print(f"  {'✓ PASS' if max(err1, err2) < 1e-3 else '✗ FAIL'}")

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Output match: {rel_err:.2e} → {'PASS' if rel_err < 0.01 else 'FAIL'}")
print(f"Reversible:   {max(err1, err2):.2e} → {'PASS' if max(err1, err2) < 1e-3 else 'FAIL'}")
print("="*70)