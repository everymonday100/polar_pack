#!/usr/bin/env python3
"""
Reversible Quaternion Block (RevNet-style) for Stage 2.

KEY FIX: quaternion linear layers amplify norm by ~2*sqrt(in_q) because
unit-quaternion elements preserve per-element norm but matrix-mult sums
in_q terms across 4 components. We apply Xavier-style output scaling to
keep activations O(1) and make the inverse numerically stable.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from quaternion_linear_unit import QuaternionLinearUnit


class QuaternionFFNLinear(nn.Module):
    """Linear-only quaternion FFN with norm-preserving output scaling."""
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.fc1 = QuaternionLinearUnit(embed_dim, ff_dim)
        self.fc2 = QuaternionLinearUnit(ff_dim, embed_dim)
        # Compensate the ~2*sqrt(in_q) amplification of each quaternion linear
        self.scale1 = 1.0 / (2.0 * math.sqrt(embed_dim // 4))
        self.scale2 = 1.0 / (2.0 * math.sqrt(ff_dim // 4))

    def forward(self, x):
        x = self.fc1(x) * self.scale1
        x = self.fc2(x) * self.scale2
        return x


class QuaternionFFN(nn.Module):
    """GELU quaternion FFN with norm-preserving output scaling."""
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.fc1 = QuaternionLinearUnit(embed_dim, ff_dim)
        self.fc2 = QuaternionLinearUnit(ff_dim, embed_dim)
        self.scale1 = 1.0 / (2.0 * math.sqrt(embed_dim // 4))
        self.scale2 = 1.0 / (2.0 * math.sqrt(ff_dim // 4))

    def forward(self, x):
        x = F.gelu(self.fc1(x) * self.scale1)
        x = self.fc2(x) * self.scale2
        return x


class ReversibleQuaternionBlockLinear(nn.Module):
    """RevNet block, LINEAR FFN, near-identity (alpha) for stable inverse."""
    def __init__(self, embed_dim, ff_dim, alpha=1.0):
        super().__init__()
        self.F = QuaternionFFNLinear(embed_dim, ff_dim)
        self.G = QuaternionFFNLinear(embed_dim, ff_dim)
        self.alpha = alpha

    def forward(self, x1, x2):
        y1 = x1 + self.alpha * self.F(x2)
        y2 = x2 + self.alpha * self.G(y1)
        return y1, y2

    def inverse(self, y1, y2):
        x2 = y2 - self.alpha * self.G(y1)
        x1 = y1 - self.alpha * self.F(x2)
        return x1, x2


class ReversibleQuaternionBlock(nn.Module):
    """RevNet block, GELU FFN, near-identity (alpha)."""
    def __init__(self, embed_dim, ff_dim, alpha=1.0):
        super().__init__()
        self.F = QuaternionFFN(embed_dim, ff_dim)
        self.G = QuaternionFFN(embed_dim, ff_dim)
        self.alpha = alpha

    def forward(self, x1, x2):
        y1 = x1 + self.alpha * self.F(x2)
        y2 = x2 + self.alpha * self.G(y1)
        return y1, y2

    def inverse(self, y1, y2):
        x2 = y2 - self.alpha * self.G(y1)
        x1 = y1 - self.alpha * self.F(x2)
        return x1, x2


class ReversibleQuaternionTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, ff_dim, num_layers,
                 linear=False, alpha=1.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim // 4)
        self.embed_dim = embed_dim
        block_cls = ReversibleQuaternionBlockLinear if linear else ReversibleQuaternionBlock
        self.blocks = nn.ModuleList([
            block_cls(embed_dim, ff_dim, alpha) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids).repeat(1, 1, 4)
        x1, x2 = x, x.clone()
        for block in self.blocks:
            x1, x2 = block(x1, x2)
        return self.out_proj(x1 + x2)

    def forward_with_intermediates(self, input_ids):
        x = self.embed(input_ids).repeat(1, 1, 4)
        x1, x2 = x, x.clone()
        intermediates = [(x1, x2)]
        for block in self.blocks:
            x1, x2 = block(x1, x2)
            intermediates.append((x1, x2))
        return intermediates, x1, x2

    def backward_recompute(self, y1_final, y2_final):
        y1, y2 = y1_final, y2_final
        for block in reversed(self.blocks):
            y1, y2 = block.inverse(y1, y2)
        return y1, y2