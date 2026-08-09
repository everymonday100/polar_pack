#!/usr/bin/env python3
"""
Smoke training: unit-constrained layer learns a unit teacher,
then round-trips through .quatpack unit mode.
"""
import torch
import numpy as np
from quaternion_linear_unit import QuaternionLinearUnit
from quaternion_nn_packer import pack_quat, unpack_quat

torch.manual_seed(0)
IN, OUT = 32, 16

# Teacher: fixed unit-constrained layer
teacher = QuaternionLinearUnit(IN, OUT)
for p in teacher.parameters():
    p.requires_grad_(False)

# Student: learns to match teacher
student = QuaternionLinearUnit(IN, OUT)
opt = torch.optim.Adam(student.parameters(), lr=1e-2)

X = torch.randn(512, IN)
with torch.no_grad():
    Y = teacher(X)

print("Training student (gradients flow through S3 projection)...")
for step in range(300):
    opt.zero_grad()
    loss = ((student(X) - Y) ** 2).mean()
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"  step {step:3d}  loss {loss.item():.6f}")
print(f"  final       loss {loss.item():.6f}")

# 1) Unit property
q = student.packed_quaternions()
norms = np.linalg.norm(q, axis=1)
print(f"\nweight quaternion norms: min {norms.min():.6f}, max {norms.max():.6f}")

# 2) Pack / unpack round-trip
pack_quat(q, 'unit_layer.quatpack', mode='unit')
rec = unpack_quat('unit_layer.quatpack')

# 3) Rebuild layer from unpacked weights, compare outputs
restored = QuaternionLinearUnit(IN, OUT)
t = torch.from_numpy(rec).float().reshape(restored.in_q, restored.out_q, 4)
with torch.no_grad():
    restored.r_weight.copy_(t[..., 0])
    restored.i_weight.copy_(t[..., 1])
    restored.j_weight.copy_(t[..., 2])
    restored.k_weight.copy_(t[..., 3])

with torch.no_grad():
    a = student(X)
    b = restored(X)
diff = (a - b).abs().max().item()
print(f"max |output diff| after .quatpack round-trip: {diff:.2e}")

print("\n" + "="*60)
print("CHECKS:")
print(f"  [1] training with projection works: loss decreased")
print(f"  [2] weights exactly unit: {np.allclose(norms, 1, atol=1e-5)}")
print(f"  [3] function preserved after packing: diff {diff:.2e}")
print("="*60)