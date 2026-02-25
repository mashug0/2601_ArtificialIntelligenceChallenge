"""
Head-to-Head Comparison: BDH (k-WTA) vs Standard Transformer
Generates comparison_data.json for the frontend Comparison page.

Dual-model loading:
  - Champion: Pre-trained BDH from checkpoint
  - Challenger: StandardTransformer initialized with same BDHConfig
    (random weights — we show architectural sparsity difference, not semantic quality)

Outputs:
  - metadata: Config details
  - sparsity_comparison: Sparsity % per layer for both models
  - concept_battle: "Apple Test" — noise activation levels for both models
  - visual_sample: 2D activation snapshots (Neurons x Time) for "Fog vs Stars"

Usage: python export_comparison.py
"""

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from model_new import BDH, BDHConfig
# We import the updated transformer which now returns 4 values
from baseline_transformer import StandardTransformer 

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint_final.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "static", "data", "comparison_data.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Concept lists for the "Apple Test" ---
CONCEPTS = ["apple", "king", "soldier", "river", "night"]
NOISE_WORDS = ["car", "sky", "run", "blue", "laugh", "stone", "chair", "wind"]

# --- Sentence for sparsity & visual sample ---
SPARSITY_SENTENCE = "The neural network is learning to understand language"


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


def load_bdh_model(config_override=None):
    """Load pre-trained BDH (the Champion)."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    model = BDH(config).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def create_baseline_model(config):
    """Create StandardTransformer with same config (the Challenger)."""
    model = StandardTransformer(config).to(DEVICE)
    model.eval()
    return model


def get_bdh_layer_activations(model, token_ids):
    """
    Run BDH forward and return post-k-WTA q tensors per layer.
    Returns list of numpy arrays, shape (T, total_neurons) per layer.
    """
    C = model.config
    nh, D, N = C.n_head, C.n_embd, C.n_neurons_per_head
    total_N = nh * N

    W_enc = model.encoder.permute(1, 0, 2).reshape(D, total_N)
    x_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        x = model.ln_in(model.embed(x_tensor))
        W_dec = model.decoder_weight.reshape(total_N, D)
        layers = []
        for i in range(C.n_layer):
            residual = x
            q = model.apply_kwta(F.relu(model.latent_norms_q[i](x @ W_enc)), C.top_k_fraction)
            v = model.apply_kwta(F.relu(model.latent_norms_v[i](x @ model.encoder_v.permute(1, 0, 2).reshape(D, total_N))), C.top_k_fraction)
            y = model.attn(q, q, v) @ W_dec + model.decoder_bias
            x = residual + model.drop(model.ln_out(y))
            layers.append(q[0].cpu().numpy())  # (T, total_N)
    return layers


def get_transformer_layer_activations(model, token_ids):
    """
    Run StandardTransformer forward and capture post-FFN hidden states per layer.
    Returns list of numpy arrays, shape (T, d_model) per layer.
    """
    x_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        # FIXED: Now unpacking 4 values instead of 3
        _, _, hidden_states, _ = model(x_tensor, return_hidden_states=True)
    return [h[0].cpu().numpy() for h in hidden_states]  # list of (T, d_model)


def get_bdh_single_word_activation(model, token_ids):
    """Get Layer 0 post-k-WTA activation for a single word (flattened)."""
    C = model.config
    nh, D, N = C.n_head, C.n_embd, C.n_neurons_per_head
    total_N = nh * N
    W_enc = model.encoder.permute(1, 0, 2).reshape(D, total_N)
    norm = model.latent_norms_q[0]

    t = torch.tensor(token_ids, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        emb = model.embed(t).mean(0).unsqueeze(0)
        act = model.apply_kwta(F.relu(norm(emb @ W_enc)), C.top_k_fraction).flatten()
    return act.cpu().numpy()


def get_transformer_single_word_activation(model, token_ids):
    """Get Layer 0 post-FFN hidden state for a single word (flattened)."""
    x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        # FIXED: Now unpacking 4 values instead of 3
        _, _, hidden_states, _ = model(x, return_hidden_states=True)
    # Layer 0 output, average across time if multi-token
    return hidden_states[0][0].mean(0).cpu().numpy()


def compute_sparsity_comparison(bdh_model, transformer_model, tokenizer):
    """
    Step 5: Run SPARSITY_SENTENCE through both models.
    For each layer, compute % of neurons that are exactly zero.
    """
    tokens = tokenizer.encode(SPARSITY_SENTENCE)
    if not tokens:
        return {"bdh": [], "transformer": []}

    # BDH layers: each is (T, total_N)
    bdh_layers = get_bdh_layer_activations(bdh_model, tokens)
    bdh_sparsity = []
    for layer_act in bdh_layers:
        zero_pct = float((layer_act == 0).mean()) * 100
        bdh_sparsity.append(round(zero_pct, 2))

    # Transformer layers: each is (T, d_model)
    trans_layers = get_transformer_layer_activations(transformer_model, tokens)
    trans_sparsity = []
    for layer_act in trans_layers:
        zero_pct = float((layer_act == 0).mean()) * 100
        trans_sparsity.append(round(zero_pct, 2))

    return {"bdh": bdh_sparsity, "transformer": trans_sparsity}


def compute_concept_battle(bdh_model, transformer_model, tokenizer):
    """
    Step 4: The "Apple Test" — Concept Isolation Logic.
    For each concept, find the most active neuron in Layer 0,
    then track that neuron's activation on noise words.
    """
    results = {}

    for concept in CONCEPTS:
        concept_tokens = tokenizer.encode(concept)
        if not concept_tokens:
            continue

        # BDH: get concept activation and find top neuron
        bdh_concept_act = get_bdh_single_word_activation(bdh_model, concept_tokens)
        bdh_top_neuron = int(np.argmax(bdh_concept_act))
        bdh_concept_strength = float(bdh_concept_act[bdh_top_neuron])

        # Transformer: get concept activation and find top neuron
        trans_concept_act = get_transformer_single_word_activation(transformer_model, concept_tokens)
        trans_top_neuron = int(np.argmax(np.abs(trans_concept_act)))
        trans_concept_strength = float(trans_concept_act[trans_top_neuron])

        # Track noise activations for those specific neurons
        bdh_noise = {}
        trans_noise = {}

        for noise_word in NOISE_WORDS:
            noise_tokens = tokenizer.encode(noise_word)
            if not noise_tokens:
                continue

            # BDH: check the concept neuron's response to noise
            bdh_noise_act = get_bdh_single_word_activation(bdh_model, noise_tokens)
            bdh_noise[noise_word] = round(float(bdh_noise_act[bdh_top_neuron]), 6)

            # Transformer: check the concept neuron's response to noise
            trans_noise_act = get_transformer_single_word_activation(transformer_model, noise_tokens)
            trans_noise[noise_word] = round(float(trans_noise_act[trans_top_neuron]), 6)

        results[concept] = {
            "bdh_top_neuron": bdh_top_neuron,
            "bdh_concept_strength": round(bdh_concept_strength, 4),
            "bdh_noise_activations": bdh_noise,
            "transformer_top_neuron": trans_top_neuron,
            "transformer_concept_strength": round(trans_concept_strength, 4),
            "transformer_noise_activations": trans_noise,
        }

    return results


def compute_visual_sample(bdh_model, transformer_model, tokenizer):
    """
    Step 6: 2D snapshot of activations (Neurons x Time) for a short phrase.
    "Fog vs Stars" — dense vs sparse patterns.
    We sample a fixed grid (e.g., 50 neurons) for visual clarity.
    """
    phrase = "The king walked to the river"
    tokens = tokenizer.encode(phrase)
    if not tokens:
        return {"phrase": phrase, "tokens_decoded": [], "bdh_grid": [], "transformer_grid": []}

    token_labels = [tokenizer.decode([t]) for t in tokens]
    SAMPLE_NEURONS = 50

    # BDH: Layer 0 activations, shape (T, total_N)
    bdh_layers = get_bdh_layer_activations(bdh_model, tokens)
    bdh_l0 = bdh_layers[0]  # (T, total_N)
    # Sample evenly spaced neurons
    n_total_bdh = bdh_l0.shape[1]
    bdh_indices = np.linspace(0, n_total_bdh - 1, SAMPLE_NEURONS, dtype=int)
    bdh_grid = np.round(bdh_l0[:, bdh_indices], 4).tolist()  # (T, SAMPLE_NEURONS)

    # Transformer: Layer 0 activations, shape (T, d_model)
    trans_layers = get_transformer_layer_activations(transformer_model, tokens)
    trans_l0 = trans_layers[0]  # (T, d_model)
    n_total_trans = trans_l0.shape[1]
    trans_indices = np.linspace(0, n_total_trans - 1, SAMPLE_NEURONS, dtype=int)
    trans_grid = np.round(trans_l0[:, trans_indices], 4).tolist()  # (T, SAMPLE_NEURONS)

    return {
        "phrase": phrase,
        "tokens_decoded": token_labels,
        "n_sampled_neurons": SAMPLE_NEURONS,
        "bdh_grid": bdh_grid,
        "transformer_grid": trans_grid,
    }


def main():
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = load_tokenizer()

    print(f"Loading BDH (Champion) from {CHECKPOINT_PATH}...")
    bdh_model, config = load_bdh_model()

    print(f"Creating StandardTransformer (Challenger) with same config...")
    transformer_model = create_baseline_model(config)

    print(f"\nConfig: {config.n_layer} layers, {config.n_embd} embd, {config.n_head} heads")
    print(f"BDH neurons/layer: {config.n_head * config.n_neurons_per_head}")
    print(f"Transformer d_model: {config.n_embd}")
    print(f"Device: {DEVICE}\n")

    # --- Step 5: Sparsity Comparison ---
    print("Computing sparsity comparison...")
    sparsity = compute_sparsity_comparison(bdh_model, transformer_model, tokenizer)
    print(f"  BDH sparsity per layer: {sparsity['bdh']}")
    print(f"  Transformer sparsity per layer: {sparsity['transformer']}")

    # --- Step 4: Concept Battle ---
    print("\nRunning Concept Battle (Apple Test)...")
    concept_battle = compute_concept_battle(bdh_model, transformer_model, tokenizer)
    for concept, data in concept_battle.items():
        bdh_max_noise = max(data["bdh_noise_activations"].values()) if data["bdh_noise_activations"] else 0
        trans_max_noise = max(abs(v) for v in data["transformer_noise_activations"].values()) if data["transformer_noise_activations"] else 0
        print(f"  {concept}: BDH max noise={bdh_max_noise:.4f}, Transformer max noise={trans_max_noise:.4f}")

    # --- Step 6: Visual Sample ---
    print("\nGenerating visual sample (Fog vs Stars)...")
    visual_sample = compute_visual_sample(bdh_model, transformer_model, tokenizer)
    print(f"  Phrase: '{visual_sample['phrase']}' ({len(visual_sample['tokens_decoded'])} tokens)")

    # --- Assemble output ---
    output = {
        "metadata": {
            "model_name_champion": "BDH k-WTA",
            "model_name_challenger": "Standard Transformer (GeLU)",
            "n_layers": config.n_layer,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "bdh_neurons_per_layer": config.n_head * config.n_neurons_per_head,
            "top_k_fraction": config.top_k_fraction,
            "vocab_size": config.vocab_size,
            "sparsity_sentence": SPARSITY_SENTENCE,
        },
        "sparsity_comparison": sparsity,
        "concept_battle": concept_battle,
        "visual_sample": visual_sample,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nExported to {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()