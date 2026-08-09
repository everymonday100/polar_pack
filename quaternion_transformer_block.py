#!/usr/bin/env python3
"""
Quaternion Transformer Block for Stage 2.
Attention + FFN, all quaternion, unit-constrained.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from quaternion_attention import QuaternionAttention
from quaternion_linear_unit import QuaternionLinearUnit


class QuaternionTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        
        self.attention = QuaternionAttention(embed_dim, num_heads)
        
        # FFN: two quaternion linear layers with ReLU
        self.ff1 = QuaternionLinearUnit(embed_dim, ff_dim)
        self.ff2 = QuaternionLinearUnit(ff_dim, embed_dim)
        
        # LayerNorm (real-valued, applied per component)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, embed_dim)
        """
        # Attention with residual
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ff_out = self.ff1(x)
        ff_out = F.relu(ff_out)
        ff_out = self.ff2(ff_out)
        x = self.norm2(x + ff_out)
        
        return x


class QuaternionTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        
        # Embedding (real -> quaternion: repeat 4 times)
        self.embed = nn.Embedding(vocab_size, embed_dim // 4)
        self.embed_dim = embed_dim
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            QuaternionTransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection (quaternion -> real for logits)
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def embed_to_quaternion(self, x):
        """Real embedding -> quaternion (repeat 4 times)."""
        # x: (B, S, embed_dim//4)
        # -> (B, S, embed_dim) by repeating
        return x.repeat(1, 1, 4)

    def forward(self, input_ids, mask=None):
        """
        input_ids: (batch, seq_len)
        Returns: logits (batch, seq_len, vocab_size)
        """
        # Embed
        x = self.embed(input_ids)  # (B, S, embed_dim//4)
        x = self.embed_to_quaternion(x)  # (B, S, embed_dim)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output projection
        logits = self.out_proj(x)  # (B, S, vocab_size)
        
        return logits