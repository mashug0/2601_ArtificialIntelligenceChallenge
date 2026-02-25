"""
BDH-Final Unified Backend
Bridges the BDH-2 k-WTA model with the BDH visualization frontend.
Serves: inference, activation atlas, hebbian traces, concept storage, topology.
"""

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Optional
from collections import defaultdict, Counter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model_new import BDH, BDHConfig
from studio import router as studio_router

# --- Configuration ---
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint_final.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_NEURONS = 25

# --- Initialize FastAPI ---
app = FastAPI(
    title="BDH Monosemanticity API",
    description="Unified backend for BDH visualization with k-WTA sparse architecture",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(studio_router)

# --- Global State ---
model: Optional[BDH] = None
config: Optional[BDHConfig] = None
tokenizer = None

# --- Request/Response Models ---
class InferenceRequest(BaseModel):
    text: str
    max_length: Optional[int] = 128


class LayerActivation(BaseModel):
    layer_idx: int
    sparsity: float
    sampled_indices: List[int]
    excitatory: List[List[float]]
    inhibitory: List[List[float]]


class InferenceResponse(BaseModel):
    input_text: str
    tokens: List[int]
    layer_activations: List[LayerActivation]
    metadata: Dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    vocab_size: int


# --- Startup ---
@app.on_event("startup")
async def load_model_on_startup():
    global model, config, tokenizer

    print(f"Loading model from {CHECKPOINT_PATH}...")
    print(f"Using device: {DEVICE}")

    try:
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            config = checkpoint["config"]
            model = BDH(config).to(DEVICE)
            model.load_state_dict(checkpoint["model"])
            model.eval()

            # Load tokenizer: prefer pre-trained tokenizer.json, fallback to custom, then tiktoken
            if os.path.exists(TOKENIZER_PATH):
                from tokenizers import Tokenizer as HFTokenizer

                _hf_tok = HFTokenizer.from_file(TOKENIZER_PATH)

                class _BDHTokenizer:
                    def __init__(self, t):
                        self.t = t
                        self.n_vocab = t.get_vocab_size()
                    def encode(self, text):
                        return self.t.encode(text).ids
                    def decode(self, tokens):
                        return self.t.decode(tokens if isinstance(tokens, list) else [tokens])

                tokenizer = _BDHTokenizer(_hf_tok)
                print(f"Loaded tokenizer from {TOKENIZER_PATH} (vocab: {tokenizer.n_vocab})")
            else:
                try:
                    from custom_tokenizer import get_custom_tokenizer
                    tokenizer = get_custom_tokenizer("", vocab_size=config.vocab_size)
                except Exception:
                    import tiktoken
                    tokenizer = tiktoken.get_encoding("gpt2")

            print(f"Model loaded! Layers: {config.n_layer}, Vocab: {config.vocab_size}")
        else:
            print(f"Checkpoint not found at {CHECKPOINT_PATH}")
            raise FileNotFoundError(f"No checkpoint at {CHECKPOINT_PATH}")

        print("API ready!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e


# --- Helper Functions ---
def _encode(text: str) -> List[int]:
    return tokenizer.encode(text)


def _decode(tokens) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(tokens if isinstance(tokens, list) else [tokens])
    return "?"


def _get_batch(text: str):
    t = torch.tensor(_encode(text), dtype=torch.long).unsqueeze(0).to(DEVICE)
    return t[:, : config.max_seq_len]


def _get_layer_activations(x):
    """Run forward pass and extract per-layer sparse activations."""
    C = config
    B, T = x.size()
    D, nh = C.n_embd, C.n_head
    N = C.n_neurons_per_head

    embed = model.ln_in(model.embed(x))

    W_enc = model.encoder.permute(1, 0, 2).reshape(D, nh * N)
    W_enc_v = model.encoder_v.permute(1, 0, 2).reshape(D, nh * N)
    W_dec = model.decoder_weight.reshape(nh * N, D)

    layer_acts = []
    curr = embed

    for i in range(C.n_layer):
        residual = curr
        q = model.apply_kwta(F.relu(model.latent_norms_q[i](curr @ W_enc)), C.top_k_fraction)
        v = model.apply_kwta(F.relu(model.latent_norms_v[i](curr @ W_enc_v)), C.top_k_fraction)

        # q is the excitatory signal, v is the inhibitory/value signal
        q_np = q[0].detach().cpu().numpy()
        v_np = v[0].detach().cpu().numpy()

        # Sparsity
        sparsity = float((q_np == 0).mean())

        # Find top-k most active neurons across all tokens
        mean_act = q_np.mean(axis=0)
        top_indices = np.argsort(mean_act)[-TOP_K_NEURONS:]

        layer_acts.append({
            "layer_idx": i,
            "sparsity": sparsity,
            "sampled_indices": top_indices.tolist(),
            "excitatory": q_np[:, top_indices].tolist(),
            "inhibitory": v_np[:, top_indices].tolist(),
        })

        # Continue forward
        y_attn = model.attn(q, q, v) @ W_dec + model.decoder_bias
        curr = residual + model.drop(model.ln_out(y_attn))

    return layer_acts


# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=DEVICE,
        vocab_size=config.vocab_size if config else 0,
    )


@app.get("/vocabulary")
async def get_vocabulary():
    """Build a vocabulary mapping for the frontend."""
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")

    # For BPE tokenizers, get the full vocabulary
    vocab = {}
    inv_vocab = {}

    try:
        # Get full vocabulary from HF tokenizer
        if hasattr(tokenizer, 't') and hasattr(tokenizer.t, 'get_vocab'):
            full_vocab = tokenizer.t.get_vocab()
            for token_str, token_id in full_vocab.items():
                vocab[token_str] = token_id
                inv_vocab[str(token_id)] = token_str
        else:
            # Fallback: build vocabulary by decoding each token id
            for i in range(config.vocab_size):
                try:
                    decoded = _decode([i])
                    if decoded and decoded != "?":
                        inv_vocab[str(i)] = decoded
                except Exception:
                    pass
    except Exception as e:
        print(f"Warning: Could not build full vocabulary: {e}")
        # Fallback to limited ASCII mapping
        for i in range(32, 127):
            c = chr(i)
            try:
                ids = _encode(c)
                if ids:
                    vocab[c] = ids[0]
                    inv_vocab[str(ids[0])] = c
            except Exception:
                pass

    return {
        "char_to_int": vocab,
        "int_to_char": inv_vocab,
        "vocab_size": config.vocab_size,
    }


@app.post("/api/infer", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = request.text[: request.max_length]
    tokens = _encode(text)
    if not tokens:
        raise HTTPException(status_code=400, detail="Could not tokenize input")

    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
    x = x[:, : config.max_seq_len]

    with torch.no_grad():
        layer_activations = _get_layer_activations(x)

    return InferenceResponse(
        input_text=text,
        tokens=tokens[: config.max_seq_len],
        layer_activations=[LayerActivation(**la) for la in layer_activations],
        metadata={
            "model_name": "BDH k-WTA",
            "n_layers": config.n_layer,
            "n_neurons": config.n_head * config.n_neurons_per_head,
            "d_model": config.n_embd,
            "vocab_size": config.vocab_size,
            "device": DEVICE,
        },
    )


@app.get("/specialists")
async def get_specialists():
    """Identify specialist neurons by probing with individual words."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    specialists = []
    probe_words = [
        "the", "and", "of", "to", "in", "a", "is", "that", "for", "it",
        "was", "on", "are", "as", "with", "his", "they", "at", "be", "this",
        "have", "from", "or", "had", "by", "not", "but", "what", "all", "were",
    ]

    C = config
    nh, D, N = C.n_head, C.n_embd, C.n_neurons_per_head
    W_enc = model.encoder.permute(1, 0, 2).reshape(D, nh * N)
    norm = model.latent_norms_q[0]

    with torch.no_grad():
        for word in probe_words:
            try:
                t = torch.tensor(_encode(word)).to(DEVICE)
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
                        "activation_strength": float(top_val),
                        "selectivity": 0.8,
                    })
            except Exception:
                continue

    return {"specialists": specialists}


# ========================================================================
# NEW ENDPOINTS: Activation Atlas, Hebbian Trace, Concept Storage, Topology
# ========================================================================

@app.get("/api/activation-atlas")
async def get_activation_atlas():
    """Generate monosemantic neuron atlas - maps neurons to semantic concepts."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    scan_words = [
        "soldier", "army", "general", "gun", "war", "peace", "love", "hate",
        "winter", "snow", "sun", "sky", "room", "door", "house", "tree",
        "flower", "river", "mountain", "king", "queen", "prince", "enemy",
        "friend", "death", "life", "night", "day", "light", "dark",
        "smile", "cry", "laugh", "walk", "run", "fire", "water", "earth",
        "wind", "sword", "shield", "blood", "heart", "mind", "soul",
        "child", "mother", "father", "battle", "victory",
    ]

    C = config
    nh, D, N = C.n_head, C.n_embd, C.n_neurons_per_head
    W_enc = model.encoder.permute(1, 0, 2).reshape(D, nh * N)
    norm = model.latent_norms_q[0]

    neuron_triggers = defaultdict(list)

    with torch.no_grad():
        for w in scan_words:
            try:
                t = torch.tensor(_encode(w)).to(DEVICE)
                if len(t) == 0:
                    continue
                emb = model.embed(t).mean(0).unsqueeze(0)
                act = model.apply_kwta(F.relu(norm(emb @ W_enc)), C.top_k_fraction).flatten()
                indices = torch.topk(act, k=10).indices.tolist()
                for idx in indices:
                    val = act[idx].item()
                    if val > 0:
                        neuron_triggers[idx].append((w, val))
            except Exception:
                continue

    # Diversity filter
    concept_cap = 8
    filled_counts = Counter()
    atlas = {}

    for n_idx, triggers in neuron_triggers.items():
        if not triggers:
            continue
        triggers.sort(key=lambda x: x[1], reverse=True)
        chosen_concept = triggers[0][0]
        for w, val in triggers:
            if filled_counts[w] < concept_cap:
                chosen_concept = w
                break
        filled_counts[chosen_concept] += 1
        atlas[int(n_idx)] = {
            "concept": chosen_concept,
            "triggers": [t[0] for t in triggers[:5]],
            "score": float(triggers[0][1]),
        }

    return {"atlas": atlas, "total_neurons_mapped": len(atlas)}


@app.get("/api/hebbian-trace")
async def get_hebbian_trace():
    """Generate frame-by-frame attention matrix animation showing synapse strengthening."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = "The general ordered the army to march."
    x = _get_batch(text)
    tokens_decoded = []
    for t_id in x[0]:
        tokens_decoded.append(_decode([t_id.item()]))

    C = config
    nh, D, N = C.n_head, C.n_embd, C.n_neurons_per_head

    with torch.no_grad():
        embed = model.ln_in(model.embed(x))
        W_enc = model.encoder.permute(1, 0, 2).reshape(D, nh * N)
        norm_q = model.latent_norms_q[0]

        q = model.apply_kwta(F.relu(norm_q(embed @ W_enc)), C.top_k_fraction)

        B, T_total, _ = q.size()
        q_heads = q.view(B, T_total, nh, N).transpose(1, 2)
        q_rot = model.attn.rope(q_heads)

        frames = []
        for t in range(1, T_total + 1):
            curr_q = q_rot[:, :, :t, :]
            raw_scores = (curr_q @ curr_q.transpose(-2, -1)) * (1.0 / (N ** 0.5))
            attn_matrix = F.softmax(raw_scores, dim=-1)[0, 0].detach().cpu().numpy()

            padded = np.zeros((T_total, T_total))
            padded[:t, :t] = attn_matrix

            frames.append({
                "step": t,
                "current_token": tokens_decoded[t - 1],
                "matrix": padded.tolist(),
            })

    return {"tokens": tokens_decoded, "frames": frames}


@app.get("/api/topology")
async def get_topology():
    """Generate network topology graph with community clustering."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    C = config
    nh, D, N = C.n_head, C.n_embd, C.n_neurons_per_head

    with torch.no_grad():
        W = model.encoder.permute(1, 0, 2).reshape(D, nh * N).detach().cpu()
        neuron_strength = W.norm(dim=0)
        top_indices = torch.topk(neuron_strength, min(150, nh * N)).indices.tolist()

        W_sub = W[:, top_indices]
        Sim = W_sub.T @ W_sub
        Sim.fill_diagonal_(0)

        threshold = torch.quantile(Sim.flatten(), 0.96)

        nodes = []
        links = []

        # Simple community detection via thresholding
        import networkx as nx

        G = nx.Graph()
        rows, cols = torch.where(Sim > threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            if r < c:
                G.add_edge(top_indices[r], top_indices[c], weight=float(Sim[r, c]))

        try:
            communities = list(nx.community.greedy_modularity_communities(G))
            node_groups = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_groups[node] = i + 1
        except Exception:
            node_groups = {n: 1 for n in top_indices}

        for idx in top_indices:
            nodes.append({
                "id": f"N{idx}",
                "group": node_groups.get(idx, 1),
                "val": float(neuron_strength[idx]),
            })

        for r, c in zip(rows.tolist(), cols.tolist()):
            if r < c:
                links.append({
                    "source": f"N{top_indices[r]}",
                    "target": f"N{top_indices[c]}",
                    "value": float(Sim[r, c]),
                })

    return {"nodes": nodes, "links": links}


@app.get("/api/concept-memory")
async def get_concept_memory():
    """Show how concepts are stored across layers - activation persistence and drift."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    probe_concepts = [
        ("soldier", "army", "war"),
        ("love", "heart", "peace"),
        ("night", "dark", "moon"),
        ("king", "queen", "prince"),
        ("fire", "light", "sun"),
    ]

    C = config
    nh, D, N = C.n_head, C.n_embd, C.n_neurons_per_head
    W_enc = model.encoder.permute(1, 0, 2).reshape(D, nh * N)
    W_enc_v = model.encoder_v.permute(1, 0, 2).reshape(D, nh * N)
    W_dec = model.decoder_weight.reshape(nh * N, D)

    concept_traces = []

    with torch.no_grad():
        for group in probe_concepts:
            group_data = {"concepts": list(group), "layers": []}

            # Encode the concept sentence
            sentence = " ".join(group)
            x = _get_batch(sentence)
            embed = model.ln_in(model.embed(x))
            curr = embed

            for layer_i in range(C.n_layer):
                residual = curr
                q = model.apply_kwta(F.relu(model.latent_norms_q[layer_i](curr @ W_enc)), C.top_k_fraction)
                v = model.apply_kwta(F.relu(model.latent_norms_v[layer_i](curr @ W_enc_v)), C.top_k_fraction)

                q_flat = q[0].mean(dim=0)  # Average across tokens
                active_mask = (q_flat > 0).float()
                active_count = int(active_mask.sum().item())
                top_neurons = torch.topk(q_flat, min(20, active_count) if active_count > 0 else 1)

                layer_data = {
                    "layer": layer_i,
                    "active_neurons": active_count,
                    "total_neurons": nh * N,
                    "sparsity": float((q_flat == 0).float().mean().item()),
                    "top_neuron_ids": top_neurons.indices.tolist(),
                    "top_neuron_values": [round(v, 4) for v in top_neurons.values.tolist()],
                    "mean_activation": float(q_flat[q_flat > 0].mean().item()) if active_count > 0 else 0,
                }
                group_data["layers"].append(layer_data)

                y_attn = model.attn(q, q, v) @ W_dec + model.decoder_bias
                curr = residual + model.drop(model.ln_out(y_attn))

            concept_traces.append(group_data)

    # Also compute semantic similarity across concepts
    similarities = []
    with torch.no_grad():
        norm = model.latent_norms_q[0]
        word_vecs = {}
        all_words = set()
        for group in probe_concepts:
            all_words.update(group)

        for w in all_words:
            try:
                t = torch.tensor(_encode(w)).to(DEVICE)
                if len(t) == 0:
                    continue
                emb = model.embed(t).mean(0).unsqueeze(0)
                vec = model.apply_kwta(F.relu(norm(emb @ W_enc)), C.top_k_fraction).flatten()
                word_vecs[w] = vec
            except Exception:
                continue

        words_list = sorted(word_vecs.keys())
        for i, w1 in enumerate(words_list):
            for j, w2 in enumerate(words_list):
                if i < j:
                    sim = F.cosine_similarity(
                        word_vecs[w1].unsqueeze(0), word_vecs[w2].unsqueeze(0)
                    ).item()
                    similarities.append({"word1": w1, "word2": w2, "similarity": round(sim, 4)})

    return {
        "concept_traces": concept_traces,
        "similarities": similarities,
    }


@app.get("/api/sparse-brain")
async def get_sparse_brain():
    """Generate sparse brain activation heatmap with forced ranking."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = "The general stood by the window and looked at the dark sky while the army marched to the river."
    x = _get_batch(text)

    C = config
    nh, D, N = C.n_head, C.n_embd, C.n_neurons_per_head

    with torch.no_grad():
        embed = model.ln_in(model.embed(x))
        W_enc = model.encoder.permute(1, 0, 2).reshape(D, nh * N)
        norm = model.latent_norms_q[0]

        latent = F.relu(norm(embed @ W_enc))
        sparse_act = model.apply_kwta(latent, C.top_k_fraction)[0].detach().cpu().numpy()

    tokens_decoded = [_decode([t.item()]) for t in x[0]]

    visual_k = max(10, int(sparse_act.shape[1] * 0.05))
    heatmap_data = []

    for t_step in range(len(tokens_decoded)):
        step_activations = sparse_act[t_step]
        top_indices = np.argsort(step_activations)[-visual_k:]
        for n_idx in top_indices:
            val = float(step_activations[n_idx])
            if val > 0:
                heatmap_data.append({
                    "token": tokens_decoded[t_step],
                    "step": t_step,
                    "neuron_id": int(n_idx),
                    "activation": round(val, 4),
                })

    return {"tokens": tokens_decoded, "activations": heatmap_data}


# ── Noise Trace ──────────────────────────────────────────────────────────────

@app.get("/api/noise-trace")
async def get_noise_trace(word: str = "king", n_passes: int = 5000, noise_scale: float = 3.0):
    """
    Run N forward passes on a single word with increasing Gaussian noise added to the
    input embedding. For each pass, return the activation value of the word's top neuron
    (identified on the clean pass). This gives a real spike-then-scatter trace from the
    actual BDH model — not a simulated curve.

    Returns:
      word: the queried word
      top_neuron: neuron index that fired strongest on clean pass
      clean_activation: baseline strength (no noise)
      trace: list of {pass_idx, noise_level, activation} — one entry per pass
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    C = config
    D, nh, N = C.n_embd, C.n_head, C.n_neurons_per_head

    tokens = _encode(word.strip())
    if not tokens:
        raise HTTPException(status_code=400, detail=f"Cannot tokenize word: {word!r}")

    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

    W_enc = model.encoder.permute(1, 0, 2).reshape(D, nh * N)
    norm_q = model.latent_norms_q[0]   # Use layer-0 q neurons

    with torch.no_grad():
        # ── Clean pass: identify top neuron ──────────────────────────────────
        clean_embed = model.ln_in(model.embed(x))        # [1, T, D]
        clean_q = model.apply_kwta(
            F.relu(norm_q(clean_embed @ W_enc)), C.top_k_fraction
        )                                                # [1, T, nh*N]
        # Average across tokens to get per-neuron mean activation
        clean_mean = clean_q[0].mean(dim=0)              # [nh*N]
        top_neuron = int(clean_mean.argmax().item())
        clean_activation = float(clean_mean[top_neuron].item())

        # ── Noisy passes: ramp noise from 0 → noise_scale over n_passes ──────
        trace = []
        for i in range(n_passes):
            # Noise level ramps linearly so we see a gradual decay effect
            level = (i / max(n_passes - 1, 1)) * noise_scale
            noise = torch.randn_like(clean_embed) * level
            noisy_embed = clean_embed + noise

            q = model.apply_kwta(
                F.relu(norm_q(noisy_embed @ W_enc)), C.top_k_fraction
            )
            act = float(q[0].mean(dim=0)[top_neuron].item())
            trace.append({
                "pass_idx": i,
                "noise_level": round(level, 5),
                "activation": round(act, 5),
            })

    return {
        "word": word,
        "top_neuron": top_neuron,
        "clean_activation": round(clean_activation, 5),
        "trace": trace,
    }


# --- Serve built frontend (when present) for single-URL hosting ---
# Must be LAST so API routes (/vocabulary, /api/*) are matched first
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_BACKEND_DIR, "..", "build")
if os.path.isdir(_BUILD_DIR):
    app.mount("/", StaticFiles(directory=_BUILD_DIR, html=True), name="frontend")


# --- Run ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
