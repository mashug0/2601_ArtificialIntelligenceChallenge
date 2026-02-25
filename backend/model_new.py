"""
BDH - STABLE k-WTA ARCHITECTURE
Fixed: Probing bug (Dimension mismatch)
"""

import dataclasses
import math
import torch
import torch.nn.functional as F
from torch import nn

@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 8 
    vocab_size: int = 10000 
    dropout: float = 0.1
    max_seq_len: int = 128
    batch_size: int = 32
    learning_rate: float = 5e-4 
    top_k_fraction: float = 0.15 
    use_weight_tying: bool = False
    rope_theta: float = 10000.0

    @property
    def n_neurons_per_head(self) -> int:
        return self.mlp_internal_dim_multiplier * self.n_embd // self.n_head


def get_freqs(D: int, theta: float, dtype: torch.dtype) -> torch.Tensor:
    def quantize(t, q=2): return (t / q).floor() * q
    return 1.0 / (theta ** (quantize(torch.arange(0, D, 1, dtype=dtype)) / D)) / (2 * math.pi)


class SparseAttention(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.n_head = config.n_head
        self.N = config.n_neurons_per_head
        self.register_buffer('freqs', get_freqs(self.N, theta=config.rope_theta, dtype=torch.float32).view(1, 1, 1, self.N))

    def rope(self, v):
        B, H, T, N = v.shape
        r_phases = torch.arange(0, T, device=v.device, dtype=torch.float32).view(1, 1, -1, 1) * self.freqs
        c, s = torch.cos((r_phases % 1) * (2 * math.pi)).to(v.dtype), torch.sin((r_phases % 1) * (2 * math.pi)).to(v.dtype)
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(B, H, T, N)
        return v * c + v_rot * s

    def forward(self, Q, K, V):
        B, T, _ = Q.size()
        q = self.rope(Q.view(B, T, self.n_head, self.N).transpose(1, 2))
        k = self.rope(K.view(B, T, self.n_head, self.N).transpose(1, 2))
        v = V.view(B, T, self.n_head, self.N).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.N))
        mask = torch.tril(torch.ones(T, T, device=q.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        return y.transpose(1, 2).contiguous().view(B, T, -1)


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh, D = config.n_head, config.n_embd
        N = config.n_neurons_per_head 

        self.encoder = nn.Parameter(torch.randn(nh, D, N) * 0.02)
        self.encoder_v = nn.Parameter(torch.randn(nh, D, N) * 0.02)
        
        self.latent_norms_q = nn.ModuleList([nn.LayerNorm(nh * N) for _ in range(config.n_layer)])
        self.latent_norms_v = nn.ModuleList([nn.LayerNorm(nh * N) for _ in range(config.n_layer)])

        self.attn = SparseAttention(config)
        self.ln_in = nn.LayerNorm(D)
        self.ln_out = nn.LayerNorm(D)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        
        self.decoder_weight = nn.Parameter(torch.randn(nh, N, D) * 0.02)
        self.decoder_bias = nn.Parameter(torch.zeros(D))
        self.lm_head = nn.Linear(D, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def apply_kwta(self, x, fraction):
        """
        Robust k-WTA that handles any tensor shape (1D, 2D, or 3D)
        """
        k = int(x.shape[-1] * fraction)
        # Sort along the last dimension (neurons)
        vals, _ = torch.topk(x, k, dim=-1)
        # Get the k-th largest value to use as threshold
        # We slice [..., -1:] to keep the rank (dimensions) intact
        threshold = vals[..., -1:] 
        return x * (x >= threshold).float()

    def forward(self, idx, targets=None, return_diagnostics=False, return_hidden_states=False):
        C = self.config
        B, T = idx.size()
        D, nh = C.n_embd, C.n_head
        N = C.n_neurons_per_head

        x = self.ln_in(self.embed(idx))
        diagnostics = {'sparsities': []} if return_diagnostics else None
        hidden_states = [] if return_hidden_states else None

        W_enc = self.encoder.permute(1, 0, 2).reshape(D, nh * N)
        W_enc_v = self.encoder_v.permute(1, 0, 2).reshape(D, nh * N)
        W_dec = self.decoder_weight.reshape(nh * N, D)

        for i in range(C.n_layer):
            residual = x

            # Expansion & Sparsity
            q = self.apply_kwta(F.relu(self.latent_norms_q[i](x @ W_enc)), C.top_k_fraction)
            v = self.apply_kwta(F.relu(self.latent_norms_v[i](x @ W_enc_v)), C.top_k_fraction)

            # Attention & Reconstruction
            y = self.attn(q, q, v) @ W_dec + self.decoder_bias
            x = residual + self.drop(self.ln_out(y))

            if return_diagnostics:
                diagnostics['sparsities'].append((q == 0).float().mean().item())

            if return_hidden_states:
                # Capture post-k-WTA sparse q and v tensors
                hidden_states.append({'q': q.detach(), 'v': v.detach()})

        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        third = hidden_states if return_hidden_states else diagnostics
        return logits, loss, third

    def semantic_probe(self, word_pairs, tokenizer):
        results = {}
        self.eval()
        nh, D, N = self.encoder.shape
        W_enc = self.encoder.permute(1, 0, 2).reshape(D, nh * N)
        norm = self.latent_norms_q[0]

        for w1, w2 in word_pairs:
            try:
                t1 = torch.tensor(tokenizer.encode(w1)).to(self.embed.weight.device)
                t2 = torch.tensor(tokenizer.encode(w2)).to(self.embed.weight.device)
                
                # Expand dimensions to [1, 1, D] for processing
                e1 = self.embed(t1).mean(0).unsqueeze(0).unsqueeze(0)
                e2 = self.embed(t2).mean(0).unsqueeze(0).unsqueeze(0)
                
                # Forward pass through Layer 0 logic
                # Now apply_kwta handles this 3D input correctly
                v1 = self.apply_kwta(F.relu(norm(e1 @ W_enc)), self.config.top_k_fraction)
                v2 = self.apply_kwta(F.relu(norm(e2 @ W_enc)), self.config.top_k_fraction)
                
                sim = F.cosine_similarity(v1.flatten().unsqueeze(0), v2.flatten().unsqueeze(0)).item()
                results[f"{w1}-{w2}"] = sim
            except Exception as e:
                # print(e) # Uncomment to debug probe errors
                results[f"{w1}-{w2}"] = 0.0
        return results

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits, _, _ = self(idx[:, -self.config.max_seq_len:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        return idx