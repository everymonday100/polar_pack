#!/usr/bin/env python3
"""Diagnose WHERE the explosion happens: forward or inverse."""
import torch
from quaternion_reversible_block import ReversibleQuaternionTransformer

torch.manual_seed(42)
VOCAB_SIZE, EMBED_DIM, FF_DIM, NUM_LAYERS = 100, 32, 64, 10
B, S = 4, 16

model = ReversibleQuaternionTransformer(VOCAB_SIZE, EMBED_DIM, FF_DIM, NUM_LAYERS, linear=True)
model.eval()
input_ids = torch.randint(0, VOCAB_SIZE, (B, S))

with torch.no_grad():
    x = model.embed(input_ids).repeat(1, 1, 4)
    x1, x2 = x, x.clone()

    print(f"{'layer':<8}{'||x1||':<14}{'||x2||':<14}{'growth':<10}")
    prev_norm = x1.norm().item()
    for i, block in enumerate(model.blocks):
        x1, x2 = block(x1, x2)
        n1, n2 = x1.norm().item(), x2.norm().item()
        print(f"{i:<8}{n1:<14.4e}{n2:<14.4e}{n1/prev_norm:<10.3f}")
        prev_norm = n1

    print(f"\nFinal ||x1|| = {x1.norm().item():.4e}")
    print("Если рост ~2x за слой → активации взрываются в forward,")
    print("и inverse делает catastrophic cancellation.")