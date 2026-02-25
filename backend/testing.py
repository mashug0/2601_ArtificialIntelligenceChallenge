"""
BDH SHOWCASE SUITE - "The Fail-Safe Data Factory"
Generates visualization-ready data with FORCED RANKING to guarantee
visible pulses and populated graphs, even for highly stable models.

OUTPUTS (in bdh_vis_data/):
1. sparse_brain.json: Guaranteed dynamic pulses (Top 5% forced).
2. topology_graph.json: Network graph with community clustering.
3. neuron_atlas.json: Diverse concept map (Stop-words capped).
4. hebbian_trace.json: Frame-by-frame animation of synapse strengthening.
"""

import torch
import torch.nn.functional as F
import json
import os
import numpy as np
import networkx as nx 
from collections import defaultdict, Counter
from tqdm import tqdm
from model_new import BDH, BDHConfig
from custom_tokenizer import get_custom_tokenizer

# --- CONFIG ---
CHECKPOINT_PATH = "checkpoint_final.pt"
OUTPUT_DIR = "bdh_vis_data"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)

class BDH_Showcase:
    def __init__(self, checkpoint_path):
        print(f"🔬 Initializing Showcase Factory on {DEVICE}...")
        # Load with weights_only=False to support older pytorch versions
        self.checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        self.config = self.checkpoint['config']
        self.model = BDH(self.config).to(DEVICE)
        self.model.load_state_dict(self.checkpoint['model'])
        self.model.eval()
        
        try:
            self.tokenizer = get_custom_tokenizer("", vocab_size=self.config.vocab_size)
        except:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("gpt2")
            
    def _get_batch(self, text):
        t = torch.tensor(self.tokenizer.encode(text), dtype=torch.long).unsqueeze(0).to(DEVICE)
        return t[:, :self.config.max_seq_len]

    # =========================================================================
    # 1. SPARSE BRAIN: Forced Ranking
    # FIX: Instead of a hard threshold, we take the Top-K activations per time step.
    # This guarantees dots will appear.
    # =========================================================================
    def generate_dynamic_sparsity(self):
        print("\n🧪 [1/4] Generating 'Sparse Brain' (Forced Ranking)...")
        text = "The general stood by the window and looked at the dark sky while the army marched to the river."
        x = self._get_batch(text)
        
        C = self.model.config
        nh, D, N = C.n_head, C.n_embd, C.n_neurons_per_head
        
        # Forward pass Layer 0
        embed = self.model.ln_in(self.model.embed(x))
        W_enc = self.model.encoder.permute(1, 0, 2).reshape(D, nh * N)
        norm = self.model.latent_norms_q[0]
        
        latent = F.relu(norm(embed @ W_enc))
        # Raw sparse activations [Batch, Time, Neurons]
        sparse_act = self.model.apply_kwta(latent, C.top_k_fraction)[0].detach().cpu().numpy()
        
        heatmap_data = []
        tokens = [self.tokenizer.decode([t.item()]) for t in x[0]]
        
        # Determine how many "dots" we want per column (Time Step)
        # 15% sparsity of ~4000 neurons is too dense for visual. Let's show top 5% purely for visual clarity.
        visual_k = int(sparse_act.shape[1] * 0.05) 
        if visual_k < 10: visual_k = 10 # Ensure at least 10 dots
        
        for t_step in range(len(tokens)):
            step_activations = sparse_act[t_step]
            
            # Find indices of the Top-K highest values for this specific token
            # This ignores the "Static Hub" problem because we rank relative to NOW.
            top_indices = np.argsort(step_activations)[-visual_k:]
            
            for n_idx in top_indices:
                val = float(step_activations[n_idx])
                if val > 0: # Only plot if actually active
                    heatmap_data.append({
                        "token": tokens[t_step],
                        "step": t_step,
                        "neuron_id": int(n_idx),
                        "activation": round(val, 4)
                    })
                
        with open(f"{OUTPUT_DIR}/sparse_brain.json", "w") as f:
            json.dump({"tokens": tokens, "activations": heatmap_data}, f)
        print(f"   -> Saved {len(heatmap_data)} pulses. (Guaranteed visible via Forced Ranking).")

    # =========================================================================
    # 2. GRAPH BRAIN: Clustered Topology
    # FIX: Uses greedy modularity for colors.
    # =========================================================================
    def generate_clustered_topology(self):
        print("\n🧪 [2/4] Extracting 'Graph Brain' with Clusters...")
        
        C = self.model.config
        nh, D, N = C.n_head, C.n_embd, C.n_neurons_per_head
        W = self.model.encoder.permute(1, 0, 2).reshape(D, nh * N).detach().cpu()
        
        # Top 150 Hubs
        neuron_strength = W.norm(dim=0)
        top_indices = torch.topk(neuron_strength, 150).indices.tolist()
        
        # Build Adjacency Matrix
        W_sub = W[:, top_indices]
        Sim = W_sub.T @ W_sub 
        Sim.fill_diagonal_(0)
        
        # Build NetworkX Graph
        G = nx.Graph()
        threshold = torch.quantile(Sim, 0.96)
        rows, cols = torch.where(Sim > threshold)
        
        for r, c in zip(rows.tolist(), cols.tolist()):
            if r < c:
                G.add_edge(top_indices[r], top_indices[c], weight=float(Sim[r, c]))
                
        # Detect Communities
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
            node_groups = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_groups[node] = i + 1
        except:
            node_groups = {n: 1 for n in top_indices}

        nodes = []
        links = []
        
        for idx in top_indices:
            nodes.append({
                "id": f"N{idx}", 
                "group": node_groups.get(idx, 1), 
                "val": float(neuron_strength[idx])
            })
            
        for r, c in zip(rows.tolist(), cols.tolist()):
            if r < c:
                links.append({
                    "source": f"N{top_indices[r]}",
                    "target": f"N{top_indices[c]}",
                    "value": float(Sim[r, c])
                })

        with open(f"{OUTPUT_DIR}/topology_graph.json", "w") as f:
            json.dump({"nodes": nodes, "links": links}, f, indent=2)
        print(f"   -> Graph saved: {len(nodes)} nodes, {len(links)} edges.")

    # =========================================================================
    # 3. MONOSEMANTIC ATLAS: Diverse Concepts
    # FIX: Caps the number of neurons assigned to generic words (like "Hate").
    # =========================================================================
    def generate_diverse_atlas(self):
        print("\n🧪 [3/4] Building Diverse Monosemantic Atlas...")
        
        scan_text = "soldier army general gun war peace love hate winter snow sun sky room door house tree flower river mountain king queen prince enemy friend death life night day light dark smile cry laugh walk run"
        words = list(set(scan_text.split()))
        
        neuron_triggers = defaultdict(list)
        
        C = self.model.config
        nh, D, N = C.n_head, C.n_embd, C.n_neurons_per_head
        W_enc = self.model.encoder.permute(1, 0, 2).reshape(D, nh * N)
        norm = self.model.latent_norms_q[0]
        
        # 1. First Pass: Collect all activations
        for w in tqdm(words, desc="Scanning"):
            t = torch.tensor(self.tokenizer.encode(w)).to(DEVICE)
            if len(t) == 0: continue
            emb = self.model.embed(t).mean(0).unsqueeze(0)
            act = self.model.apply_kwta(F.relu(norm(emb @ W_enc)), C.top_k_fraction).flatten()
            
            # Record strongest activations
            indices = torch.topk(act, k=10).indices.tolist() 
            for idx in indices:
                val = act[idx].item()
                if val > 0:
                    neuron_triggers[idx].append((w, val))

        # 2. Diversity Filter
        # A word can only be the 'Top Concept' for 8 neurons max
        concept_cap = 8
        filled_counts = Counter()
        
        atlas = {}
        for n_idx, triggers in neuron_triggers.items():
            if not triggers: continue
            
            # Sort triggers by strength
            triggers.sort(key=lambda x: x[1], reverse=True)
            
            # Find the best concept that hasn't hit the cap
            chosen_concept = triggers[0][0]
            for w, val in triggers:
                if filled_counts[w] < concept_cap:
                    chosen_concept = w
                    break
            
            filled_counts[chosen_concept] += 1
            
            atlas[int(n_idx)] = {
                "concept": chosen_concept,
                "triggers": [t[0] for t in triggers[:5]],
                "score": float(triggers[0][1])
            }
            
        with open(f"{OUTPUT_DIR}/neuron_atlas.json", "w") as f:
            json.dump(atlas, f, indent=2)
        print(f"   -> Atlas saved. (Diversity Cap: {concept_cap}).")

    # =========================================================================
    # 4. MEMORY ANIMATION: Frame-by-Frame
    # =========================================================================
    def generate_hebbian_frames(self):
        print("\n🧪 [4/4] Generating Hebbian Animation Frames...")
        
        text = "The general ordered the army to march."
        x = self._get_batch(text)
        tokens = [self.tokenizer.decode([t.item()]) for t in x[0]]
        
        C = self.model.config
        nh, D, N = C.n_head, C.n_embd, C.n_neurons_per_head
        
        embed = self.model.ln_in(self.model.embed(x))
        W_enc = self.model.encoder.permute(1, 0, 2).reshape(D, nh * N)
        norm_q = self.model.latent_norms_q[0]
        
        q = self.model.apply_kwta(F.relu(norm_q(embed @ W_enc)), C.top_k_fraction)
        
        attn_module = self.model.attn
        B, T_total, _ = q.size()
        q_heads = q.view(B, T_total, nh, N).transpose(1, 2)
        q_rot = attn_module.rope(q_heads)
        
        frames = []
        
        # Generate 1 frame per token step (Triangular growth)
        for t in range(1, T_total + 1):
            curr_q = q_rot[:, :, :t, :] 
            raw_scores = (curr_q @ curr_q.transpose(-2, -1)) * (1.0 / (N**0.5))
            attn_matrix = F.softmax(raw_scores, dim=-1)[0, 0].detach().cpu().numpy() 
            
            padded = np.zeros((T_total, T_total))
            padded[:t, :t] = attn_matrix
            
            frames.append({
                "step": t,
                "current_token": tokens[t-1],
                "matrix": padded.tolist()
            })
            
        with open(f"{OUTPUT_DIR}/hebbian_trace.json", "w") as f:
            json.dump({"tokens": tokens, "frames": frames}, f)
        print(f"   -> Saved {len(frames)} animation frames.")

# --- RUN ---
if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint '{CHECKPOINT_PATH}' not found!")
    else:
        lab = BDH_Showcase(CHECKPOINT_PATH)
        lab.generate_dynamic_sparsity()
        lab.generate_clustered_topology()
        lab.generate_diverse_atlas()
        lab.generate_hebbian_frames()
        print("\n🎉 SHOWCASE DATA READY in 'bdh_vis_data/'")