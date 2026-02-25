"""
Battle Data Exporter — "The Strawman Showdown"
Runs the same prompt through BDH and Dense Transformer side-by-side.

Key metric: "Active Load"
  - Transformer (GeLU): Guaranteed ~100% active.
    We now export ATTENTION CONNECTIONS to visualize the "messy hairball".
  - BDH (k-WTA): Only top_k_fraction survive.
    We export the sparse topology.

Usage: python export_battle.py
"""

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from model_new import BDH, BDHConfig
from baseline_transformer import StandardTransformer  # Fixed import

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint_final.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "static", "data", "battle_data.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_TEXT = "The quick brown fox jumps over the lazy dog."


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


def extract_transformer_data(model, token_ids, config):
    """
    Dense path: Run transformer, capture attention matrices.
    Returns 'active_links' for the 3D visualization to show the 'Dense Web'.
    """
    x_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    T = len(token_ids)

    with torch.no_grad():
        # Get attention maps!
        logits, loss, hidden_states, attentions = model(x_tensor, return_attention=True)

        # Also run through blocks manually just to get FFN activity % (load)
        # (This is a bit redundant but safe for existing logic)
        pos = torch.arange(0, T, device=x_tensor.device).unsqueeze(0)
        x_dummy = model.drop(model.ln_in(model.embed(x_tensor) + model.pos_embed(pos)))
        
        layer_loads = []
        for block in model.blocks:
            x_dummy, _ = block(x_dummy)
            activity = float((x_dummy.abs() > 1e-8).float().mean().item()) * 100
            layer_loads.append(round(activity, 2))

    # --- PROCESS ATTENTION FOR VISUALIZATION ---
    # attentions is list of (1, n_head, T, T)
    visual_layers = []
    
    for i, layer_att in enumerate(attentions):
        # Average across heads to get a single "connection strength" matrix
        # Shape: (T, T)
        avg_att = layer_att[0].mean(dim=0).cpu().numpy() 
        
        connections = []
        # Create a list of connections for the 3D visualizer
        # We assume A[i, j] is weight from j -> i
        # We threshold to keep the JSON size reasonable, but keep it dense enough to look "messy"
        # Threshold 0.02 means we keep any connection > 2% attention
        rows, cols = np.where(avg_att > 0.02)
        
        for r, c in zip(rows, cols):
            connections.append({
                "source": int(c), # Token index (From)
                "target": int(r), # Token index (To)
                "weight": round(float(avg_att[r, c]), 4)
            })

        visual_layers.append({
            "layer_idx": i,
            "connections": connections
        })

    return {
        "layer_loads": layer_loads,
        "avg_load": round(sum(layer_loads) / len(layer_loads), 2),
        "visual_layers": visual_layers, 
    }


def extract_bdh_data(model, token_ids, config):
    """
    Sparse path: Run BDH, capture active neurons.
    """
    C = config
    nh, D, N = C.n_head, C.n_embd, C.n_neurons_per_head
    total_N = nh * N
    T = len(token_ids)

    W_enc = model.encoder.permute(1, 0, 2).reshape(D, total_N)
    W_enc_v = model.encoder_v.permute(1, 0, 2).reshape(D, total_N)
    W_dec = model.decoder_weight.reshape(total_N, D)

    x_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        x = model.ln_in(model.embed(x_tensor))

        all_layer_acts = []  
        layer_loads = []

        for i in range(C.n_layer):
            residual = x
            q = model.apply_kwta(F.relu(model.latent_norms_q[i](x @ W_enc)), C.top_k_fraction)
            v = model.apply_kwta(F.relu(model.latent_norms_v[i](x @ W_enc_v)), C.top_k_fraction)
            y = model.attn(q, q, v) @ W_dec + model.decoder_bias
            x = residual + model.drop(model.ln_out(y))

            q_np = q[0].cpu().numpy()  # (T, total_N)
            all_layer_acts.append(q_np)
            activity = float((q_np > 0).mean()) * 100
            layer_loads.append(round(activity, 2))

    # Normalize
    global_max_act = 0.0
    for layer_act in all_layer_acts:
        m = float(layer_act.max())
        if m > global_max_act: global_max_act = m
    if global_max_act < 1e-8: global_max_act = 1.0

    # Build graph
    token_graphs = []
    for t in range(T):
        active_nodes = {}
        active_links = []
        prev_active_ids = []

        for layer_idx, layer_act in enumerate(all_layer_acts):
            token_act = layer_act[t]
            active_mask = token_act > 0
            active_indices = np.where(active_mask)[0].tolist()
            active_vals = token_act[active_mask].tolist()

            current_ids = []
            for ni, nv in zip(active_indices, active_vals):
                node_id = f"L{layer_idx}_N{ni}"
                current_ids.append(node_id)
                active_nodes[node_id] = round(float(nv) / global_max_act, 4)

            # Sparse links
            if prev_active_ids and current_ids:
                max_e = min(15, len(prev_active_ids))
                max_t = min(15, len(current_ids))
                for src in prev_active_ids[:max_e]:
                    for tgt in current_ids[:max_t]:
                        active_links.append({"source": src, "target": tgt})

            prev_active_ids = current_ids

        token_graphs.append({
            "token_idx": t,
            "n_active": len(active_nodes),
            "n_total": total_N * C.n_layer,
            "active_nodes": active_nodes,
            "active_links": active_links,
        })

    return {
        "layer_loads": layer_loads,
        "avg_load": round(sum(layer_loads) / len(layer_loads), 2),
        "total_neurons_per_layer": total_N,
        "token_graphs": token_graphs,
    }


def main():
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = load_tokenizer()

    print(f"Loading BDH (Champion) from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    bdh_model = BDH(config).to(DEVICE)
    bdh_model.load_state_dict(checkpoint["model"])
    bdh_model.eval()

    print(f"Creating StandardTransformerBaseline (Villain)...")
    transformer_model = StandardTransformer(config).to(DEVICE)
    transformer_model.eval()

    token_ids = tokenizer.encode(INPUT_TEXT)[:config.max_seq_len]
    token_labels = [tokenizer.decode([t]) for t in token_ids]

    print(f"Input: \"{INPUT_TEXT}\"")
    
    # --- Data Extraction ---
    print("\nExtracting Transformer (Dense)...")
    trans_data = extract_transformer_data(transformer_model, token_ids, config)
    
    print("\nExtracting BDH (Sparse)...")
    bdh_data = extract_bdh_data(bdh_model, token_ids, config)

    energy_savings = round(trans_data["avg_load"] - bdh_data["avg_load"], 2)

    output = {
        "metadata": {
            "input_text": INPUT_TEXT,
            "tokens": token_labels,
            "n_tokens": len(token_ids),
            "n_layers": config.n_layer,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "top_k_fraction": config.top_k_fraction,
            "bdh_neurons_per_layer": config.n_head * config.n_neurons_per_head,
        },
        "transformer_load": trans_data["avg_load"],
        "bdh_load": bdh_data["avg_load"],
        "energy_savings": energy_savings,
        "transformer_layer_loads": trans_data["layer_loads"],
        "bdh_layer_loads": bdh_data["layer_loads"],
        
        # New: List of {layer_idx, connections: [{source, target, weight}]}
        "transformer_layers": trans_data["visual_layers"], 
        
        "bdh_graph": bdh_data["token_graphs"],
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nExported to {OUTPUT_PATH}")
    print("Done!")

if __name__ == "__main__":
    main()