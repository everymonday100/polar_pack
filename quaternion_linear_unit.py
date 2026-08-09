#!/usr/bin/env python3
"""
Unit-quaternion Linear layer for Stage 2.
Each quaternion weight element is projected onto S3 before use: q -> q/|q|.
Weights become rotation-valued (reversible semantics) and natively
compatible with .quatpack unit mode.
"""
import torch
import torch.nn as nn
import math


class QuaternionLinearUnit(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        assert in_features % 4 == 0 and out_features % 4 == 0
        self.in_q = in_features // 4
        self.out_q = out_features // 4
        self.bias_on = bias

        for c in 'rijk':
            setattr(self, f'{c}_weight', nn.Parameter(
                torch.empty(self.in_q, self.out_q)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # quaternion-aware init (Gaudet & Maida style)
        std = 1.0 / math.sqrt(2 * self.in_q)
        for c in 'rijk':
            nn.init.normal_(getattr(self, f'{c}_weight'), 0, std)

    def unit_weights(self):
        """Project all quaternion elements onto S3."""
        q = torch.stack([getattr(self, f'{c}_weight') for c in 'rijk'], dim=0)
        norm = torch.sqrt((q * q).sum(dim=0, keepdim=True)) + 1e-8
        q = q / norm
        return q[0], q[1], q[2], q[3]

    def forward(self, x):
        xr, xi, xj, xk = x.chunk(4, dim=-1)
        r, i, j, k = self.unit_weights()
        # Hamilton product: out = x (x) w
        or_ = xr @ r - xi @ i - xj @ j - xk @ k
        oi  = xr @ i + xi @ r + xj @ k - xk @ j
        oj  = xr @ j - xi @ k + xj @ r + xk @ i
        ok  = xr @ k + xi @ j - xj @ i + xk @ r
        out = torch.cat([or_, oi, oj, ok], dim=-1)
        if self.bias_on:
            out = out + self.bias
        return out

    def packed_quaternions(self):
        """(N, 4) array of unit quaternion elements, layout for .quatpack."""
        r, i, j, k = self.unit_weights()
        return (torch.stack([r, i, j, k], dim=-1)
                .reshape(-1, 4).detach().cpu().numpy())