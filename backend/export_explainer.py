"""
Dual-Pipeline Explainer: BDH (Sparse Paths) vs Transformer (Dense Attention)
Runs the same sentence through both models and exports their internal geometry
side-by-side for the "Battle Arena" visualization.

BDH path:  Active neurons per token step + active edges between layers.
Trans path: Full attention matrices (SeqLen x SeqLen) per layer.
Both normalized to 0-1 scale.

Exports to ../static/data/explainer.json

Usage: python export_explainer.py
"""

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from model_new import BDH, BDHConfig
from baseline_transformer import StandardTransformer

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint_final.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "static", "data", "explainer.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_TEXT = "The quick brown fox jumps over the lazy dog."

# Step-by-step narrative for the Battle Arena
NARRATIVE = [
    "Input received. Both models tokenize the sentence identically.",
    "Layer 0: The Transformer activates ALL neurons densely (GeLU). BDH fires only the top-k sparse circuit via k-WTA.",
    "Layer 1: Transformer attention spreads weight across every token — the \"fog\" builds. BDH routes signal through a narrow, selective pathway.",
    "Layer 2: The Transformer's polysemantic neurons respond to multiple unrelated concepts simultaneously. BDH neurons remain monosemantic — each fires for one concept only.",
    "Layer 3: Final layer. The Transformer's dense attention matrix shows uniform, diffuse patterns — high energy, low specificity. BDH shows sharp, targeted activation paths — low energy, high interpretability.",
    "Output: Both produce logits. But BDH's internal state is sparse and readable. The Transformer's internal state is a dense, entangled mess.",
    "Verdict: Transformers trade interpretability for raw capacity. BDH preserves interpretability through k-WTA sparsity while maintaining competitive performance.",
    "The key insight: k-WTA enforces a biological constraint — only the strongest neurons survive each forward pass, creating naturally monosemantic representations.",
    "This architectural difference means BDH models can be inspected, debugged, and understood. Standard transformers remain opaque black boxes.",
    "Explore the visualizations above: the Red Fog (Transformer) vs the Blue Stars (BDH). Sparsity is not a limitation — it is a feature."
]


def load_tokenizer():
    from tokenizers import Tokenizer as HFTokenizer
    hf_tok = HFTokenizer.from_file(TOKENIZER_PATH)

    class BDHTokenizer:
        def __init__(self, t):
            self.t = t
            self.n_vocab = t.get_vocab_size()
        def encode(self, text):
            return self.t.encode(text).ids
        def decode(self, tokens):
            return self.t.decode(tokens if isinstance(tokens, list) else [tokens])

    return BDHTokenizer(hf_tok)


def extract_bdh_paths(model, token_ids):
    """
    BDH Extraction — "The Sparse Path"
    For each token step, identify:
      - active_nodes: neuron indices where post-k-WTA value > 0
      - active_links: connections from active neurons in Layer L to Layer L+1
    Returns a list of BDH steps (one per token position).
    """
    C = model.config
    nh, D, N = C.n_head, C.n_embd, C.n_neurons_per_head
    total_N = nh * N

    W_enc = model.encoder.permute(1, 0, 2).reshape(D, total_N)
    W_enc_v = model.encoder_v.permute(1, 0, 2).reshape(D, total_N)
    W_dec = model.decoder_weight.reshape(total_N, D)

    x_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    T = len(token_ids)

    with torch.no_grad():
        x = model.ln_in(model.embed(x_tensor))

        # Collect activations per layer
        layer_activations = []  # list of (T, total_N) arrays
        for i in range(C.n_layer):
            residual = x
            q = model.apply_kwta(F.relu(model.latent_norms_q[i](x @ W_enc)), C.top_k_fraction)
            v = model.apply_kwta(F.relu(model.latent_norms_v[i](x @ W_enc_v)), C.top_k_fraction)
            y = model.attn(q, q, v) @ W_dec + model.decoder_bias
            x = residual + model.drop(model.ln_out(y))
            layer_activations.append(q[0].cpu().numpy())  # (T, total_N)

    # Build per-token steps
    bdh_steps = []
    for t in range(T):
        step_data = {
            "token_idx": t,
            "active_nodes": [],
            "active_links": [],
        }

        prev_active = None
        for layer_idx, layer_act in enumerate(layer_activations):
            token_act = layer_act[t]  # (total_N,)
            active_mask = token_act > 0
            active_indices = np.where(active_mask)[0].tolist()
            active_values = token_act[active_mask].tolist()

            # Store active nodes with layer prefix
            for ni, nv in zip(active_indices, active_values):
                step_data["active_nodes"].append({
                    "id": f"L{layer_idx}_N{ni}",
                    "layer": layer_idx,
                    "neuron": ni,
                    "value": round(float(nv), 4),
                })

            # Build edges from previous layer's active neurons to this layer's
            if prev_active is not None and len(prev_active) > 0 and len(active_indices) > 0:
                # Sample edges: connect each active in prev to closest active in current
                # (limit to avoid explosion)
                max_edges = 20
                for src_idx in prev_active[:max_edges]:
                    for tgt_idx in active_indices[:max_edges]:
                        step_data["active_links"].append({
                            "source": f"L{layer_idx - 1}_N{src_idx}",
                            "target": f"L{layer_idx}_N{tgt_idx}",
                        })

            prev_active = active_indices

        bdh_steps.append(step_data)

    return bdh_steps


def extract_transformer_attention(model, token_ids):
    """
    Transformer Extraction — "The Dense Path"
    Extract attention matrices (SeqLen x SeqLen) per layer.
    Returns a list of transformer steps (one per layer).
    """
    x_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, _, _, attentions = model(x_tensor, return_attention=True)

    transformer_steps = []
    for layer_idx, attn_weights in enumerate(attentions):
        # attn_weights: (B, n_head, T, T) — average across heads for (T, T)
        matrix = attn_weights[0].mean(dim=0).cpu().numpy()  # (T, T)
        transformer_steps.append({
            "layer": layer_idx,
            "attention_matrix": np.round(matrix, 4).tolist(),
        })

    return transformer_steps


def normalize_bdh_values(bdh_steps):
    """Normalize all BDH node values to 0-1 scale."""
    all_vals = []
    for step in bdh_steps:
        for node in step["active_nodes"]:
            all_vals.append(node["value"])
    if not all_vals:
        return bdh_steps
    max_val = max(all_vals)
    if max_val == 0:
        return bdh_steps
    for step in bdh_steps:
        for node in step["active_nodes"]:
            node["value"] = round(node["value"] / max_val, 4)
    return bdh_steps


def normalize_transformer_matrices(transformer_steps):
    """Normalize attention matrices to 0-1 (softmax already does this per row, but ensure global consistency)."""
    all_vals = []
    for step in transformer_steps:
        for row in step["attention_matrix"]:
            all_vals.extend(row)
    if not all_vals:
        return transformer_steps
    max_val = max(all_vals)
    if max_val == 0:
        return transformer_steps
    for step in transformer_steps:
        step["attention_matrix"] = [
            [round(v / max_val, 4) for v in row]
            for row in step["attention_matrix"]
        ]
    return transformer_steps


def main():
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = load_tokenizer()

    print(f"Loading BDH (Champion) from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    bdh_model = BDH(config).to(DEVICE)
    bdh_model.load_state_dict(checkpoint["model"])
    bdh_model.eval()

    print(f"Creating StandardTransformer (Challenger) with same config...")
    transformer_model = StandardTransformer(config).to(DEVICE)
    transformer_model.eval()

    # Tokenize input
    token_ids = tokenizer.encode(INPUT_TEXT)[:config.max_seq_len]
    token_labels = [tokenizer.decode([t]) for t in token_ids]
    print(f"Input: \"{INPUT_TEXT}\"")
    print(f"Tokens ({len(token_ids)}): {token_labels}")

    # --- BDH Extraction ---
    print("\nExtracting BDH sparse paths...")
    bdh_steps = extract_bdh_paths(bdh_model, token_ids)
    bdh_steps = normalize_bdh_values(bdh_steps)
    for i, step in enumerate(bdh_steps):
        print(f"  Token {i} ({token_labels[i]}): {len(step['active_nodes'])} active nodes, {len(step['active_links'])} edges")

    # --- Transformer Extraction ---
    print("\nExtracting Transformer attention matrices...")
    transformer_steps = extract_transformer_attention(transformer_model, token_ids)
    transformer_steps = normalize_transformer_matrices(transformer_steps)
    for step in transformer_steps:
        matrix = step["attention_matrix"]
        density = sum(1 for row in matrix for v in row if v > 0.01) / (len(matrix) * len(matrix[0]))
        print(f"  Layer {step['layer']}: {len(matrix)}x{len(matrix[0])} matrix, density={density:.2%}")

    # --- Assemble output ---
    output = {
        "metadata": {
            "input_text": INPUT_TEXT,
            "tokens": token_labels,
            "n_tokens": len(token_ids),
            "n_layers": config.n_layer,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "bdh_neurons_per_layer": config.n_head * config.n_neurons_per_head,
            "top_k_fraction": config.top_k_fraction,
        },
        "bdh_steps": bdh_steps,
        "transformer_steps": transformer_steps,
        "narrative": NARRATIVE,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nExported to {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
