"""
Export activation data from BDH-2 model to static JSON for frontend.
Run this once after training to generate pre-computed visualization data.

Usage: python export_activations.py
"""

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from model_new import BDH, BDHConfig

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint_final.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "static", "data", "activations.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_NEURONS = 25

def main():
    print(f"Loading model from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    model = BDH(config).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Load tokenizer
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

    tokenizer = BDHTokenizer(hf_tok)
    print(f"Model: {config.n_layer} layers, vocab {config.vocab_size}, device {DEVICE}")

    # Build vocabulary mapping
    vocab = {}
    inv_vocab = {}
    for i in range(32, 127):
        c = chr(i)
        try:
            ids = tokenizer.encode(c)
            if ids:
                vocab[c] = ids[0]
                inv_vocab[str(ids[0])] = c
        except:
            pass
    for c in ["\n", "\t", " "]:
        try:
            ids = tokenizer.encode(c)
            if ids:
                vocab[c] = ids[0]
                inv_vocab[str(ids[0])] = c
        except:
            pass

    # Test sequences
    sequences_text = [
        "The general ordered the army to march to the river.",
        "She walked through the door and smiled at the light.",
    ]

    C = config
    nh, D, N_per = C.n_head, C.n_embd, C.n_neurons_per_head
    total_N = nh * N_per

    W_enc = model.encoder.permute(1, 0, 2).reshape(D, total_N)
    W_enc_v = model.encoder_v.permute(1, 0, 2).reshape(D, total_N)
    W_dec = model.decoder_weight.reshape(total_N, D)

    sequences = []
    for seq_i, text in enumerate(sequences_text):
        tokens = tokenizer.encode(text)[:config.max_seq_len]
        if not tokens:
            continue
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            embed = model.ln_in(model.embed(x))
            curr = embed
            layer_activations = []

            for i in range(C.n_layer):
                residual = curr
                q = model.apply_kwta(F.relu(model.latent_norms_q[i](curr @ W_enc)), C.top_k_fraction)
                v = model.apply_kwta(F.relu(model.latent_norms_v[i](curr @ W_enc_v)), C.top_k_fraction)

                q_np = q[0].detach().cpu().numpy()
                v_np = v[0].detach().cpu().numpy()

                sparsity = float((q_np == 0).mean())
                mean_act = q_np.mean(axis=0)
                top_indices = np.argsort(mean_act)[-TOP_K_NEURONS:]

                layer_activations.append({
                    "layer_idx": i,
                    "sparsity": round(sparsity, 4),
                    "sampled_indices": top_indices.tolist(),
                    "excitatory": np.round(q_np[:, top_indices], 4).tolist(),
                    "inhibitory": np.round(v_np[:, top_indices], 4).tolist(),
                })

                y_attn = model.attn(q, q, v) @ W_dec + model.decoder_bias
                curr = residual + model.drop(model.ln_out(y_attn))

        sequences.append({
            "id": f"seq_{seq_i}",
            "input_text": text,
            "tokens": tokens,
            "layer_activations": layer_activations,
        })
        print(f"  Processed: '{text[:50]}...' ({len(tokens)} tokens)")

    # Specialists
    specialists = []
    probe_words = ["the", "and", "of", "to", "in", "a", "is", "that", "for", "it",
                   "was", "on", "are", "as", "with", "his", "they", "at", "be", "this"]
    norm = model.latent_norms_q[0]

    with torch.no_grad():
        for word in probe_words:
            try:
                t = torch.tensor(tokenizer.encode(word)).to(DEVICE)
                if len(t) == 0:
                    continue
                emb = model.embed(t).mean(0).unsqueeze(0)
                act = model.apply_kwta(F.relu(norm(emb @ W_enc)), C.top_k_fraction).flatten()
                top_neuron = torch.argmax(act).item()
                top_val = act[top_neuron].item()
                if top_val > 0.3:
                    specialists.append({
                        "neuron_idx": int(top_neuron),
                        "layer": 0,
                        "trigger_char": word,
                        "activation_strength": round(float(top_val), 4),
                        "selectivity": 0.8,
                    })
            except:
                continue

    output = {
        "metadata": {
            "model_name": "BDH k-WTA",
            "n_layers": config.n_layer,
            "n_neurons": total_N,
            "d_model": config.n_embd,
            "vocab_size": config.vocab_size,
            "threshold": config.top_k_fraction,
        },
        "vocabulary": {
            "char_to_int": vocab,
            "int_to_char": inv_vocab,
        },
        "specialists": specialists,
        "sequences": sequences,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f)

    print(f"\nExported to {OUTPUT_PATH}")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Layers: {config.n_layer}")
    print(f"  Neurons/layer sampled: {TOP_K_NEURONS}")
    print(f"  Specialists: {len(specialists)}")

if __name__ == "__main__":
    main()
