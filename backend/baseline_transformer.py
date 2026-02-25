"""
Standard Transformer Baseline - The "Challenger"
Uses the same BDHConfig parameters (n_layer, n_embd, n_head) for a fair
comparison, but replaces k-WTA with GeLU and SparseAttention with standard
softmax multihead attention (no RoPE sparse constraints).
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from model_new import BDHConfig


class StandardAttention(nn.Module):
    """Standard causal multihead attention with learned positional bias (no RoPE)."""
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.n_head = config.n_head
        self.d_head = config.n_embd // config.n_head
        self.d_model = config.n_embd

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, return_weights=False):
        B, T, D = x.size()
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # (B, n_head, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Standard Scaled Dot-Product Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(y)
        
        if return_weights:
            return out, att
        return out


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block: LN -> MHA -> residual -> LN -> FFN -> residual."""
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = StandardAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        # FFN with GeLU (dense activation — the "polysemantic mess")
        ffn_dim = config.mlp_internal_dim_multiplier * config.n_embd
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, config.n_embd, bias=False),
        )
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, return_weights=False):
        if return_weights:
            attn_out, attn_weights = self.attn(self.ln1(x), return_weights=True)
            x = x + self.drop(attn_out)
        else:
            x = x + self.drop(self.attn(self.ln1(x)))
            attn_weights = None
            
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, attn_weights


class StandardTransformer(nn.Module):
    """
    Baseline dense transformer using the same BDHConfig so sizes match exactly.
    Key differences from BDH:
      - GeLU activation instead of k-WTA
      - Standard softmax attention instead of SparseAttention with RoPE
    """
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.n_embd)
        self.ln_in = nn.LayerNorm(config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.drop = nn.Dropout(config.dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_hidden_states=False, return_attention=False):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.ln_in(self.embed(idx) + self.pos_embed(pos)))

        hidden_states = [] if return_hidden_states else None
        all_attentions = [] if return_attention else None

        for block in self.blocks:
            x, att = block(x, return_weights=return_attention)
            
            if return_hidden_states:
                hidden_states.append(x.detach())
            if return_attention:
                all_attentions.append(att.detach())

        x = self.ln_out(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        
        # If we asked for attention, return it as the last element
        return logits, loss, hidden_states, all_attentions