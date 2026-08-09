#!/usr/bin/env python3
"""
Quaternion Multi-Head Attention for Stage 2.
Q, K, V are quaternions. Attention score: q ⊗ k* (Hamilton product with conjugate).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from quaternion_linear_unit import QuaternionLinearUnit
import math


class QuaternionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % 4 == 0
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V projections (quaternion, unit-constrained)
        self.q_proj = QuaternionLinearUnit(embed_dim, embed_dim, bias=False)
        self.k_proj = QuaternionLinearUnit(embed_dim, embed_dim, bias=False)
        self.v_proj = QuaternionLinearUnit(embed_dim, embed_dim, bias=False)
        self.out_proj = QuaternionLinearUnit(embed_dim, embed_dim, bias=False)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def hamilton_product(self, q1, q2):
        """Hamilton product: q1 ⊗ q2. Input: (..., 4)."""
        r1, i1, j1, k1 = q1.unbind(-1)
        r2, i2, j2, k2 = q2.unbind(-1)
        
        r = r1*r2 - i1*i2 - j1*j2 - k1*k2
        i = r1*i2 + i1*r2 + j1*k2 - k1*j2
        j = r1*j2 - i1*k2 + j1*r2 + k1*i2
        k = r1*k2 + i1*j2 - j1*i2 + k1*r2
        
        return torch.stack([r, i, j, k], dim=-1)

    def conjugate(self, q):
        """Quaternion conjugate: (r, i, j, k) -> (r, -i, -j, -k)."""
        return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, embed_dim)
        Returns: (batch, seq_len, embed_dim)
        """
        B, S, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (B, S, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape to (B, num_heads, S, head_dim)
        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Reshape head_dim to (head_dim//4, 4) for quaternion ops
        Q = Q.reshape(B, self.num_heads, S, self.head_dim // 4, 4)
        K = K.reshape(B, self.num_heads, S, self.head_dim // 4, 4)
        V = V.reshape(B, self.num_heads, S, self.head_dim // 4, 4)
        
        # Attention scores: Q ⊗ K* (Hamilton product with conjugate)
        # K*: (B, num_heads, S, head_dim//4, 4)
        K_star = self.conjugate(K)
        
        # Compute attention scores: sum over quaternion components
        # For each position pair, compute Q ⊗ K* and take real part
        # Reshape for batched computation
        Q = Q.reshape(B * self.num_heads, S, self.head_dim // 4, 4)
        K_star = K_star.reshape(B * self.num_heads, S, self.head_dim // 4, 4)
        
        # Q ⊗ K* for all pairs
        # Q: (B*H, S_q, D, 4), K*: (B*H, S_k, D, 4)
        # Expand for pairwise computation
        Q_exp = Q.unsqueeze(2)  # (B*H, S_q, 1, D, 4)
        K_exp = K_star.unsqueeze(1)  # (B*H, 1, S_k, D, 4)
        
        # Hamilton product
        qk = self.hamilton_product(Q_exp, K_exp)  # (B*H, S_q, S_k, D, 4)
        
        # Take real part and sum over D dimension
        scores = qk[..., 0].sum(dim=-1)  # (B*H, S_q, S_k)
        scores = scores * self.scale
        
        # Reshape back
        scores = scores.reshape(B, self.num_heads, S, S)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to V
        # V: (B, num_heads, S, head_dim//4, 4)
        V = V.reshape(B * self.num_heads, S, self.head_dim // 4, 4)
        
        # attn_weights: (B, num_heads, S_q, S_k)
        attn_weights = attn_weights.reshape(B * self.num_heads, S, S)
        
        # Weighted sum: (B*H, S_q, S_k) @ (B*H, S_k, D, 4) -> (B*H, S_q, D, 4)
        out = torch.einsum('bij,bjdq->bidq', attn_weights, V)
        
        # Reshape to (B, num_heads, S, head_dim)
        out = out.reshape(B, self.num_heads, S, self.head_dim)
        out = out.transpose(1, 2).reshape(B, S, self.embed_dim)
        
        # Output projection
        out = self.out_proj(out)
        
        return out