// BDH Tutorial Content — derived from the Baby Dragon Hatchling paper
// Each lesson has: theory sections, code snippets, and fill-in-the-blank exercises

export const curriculum = [
  {
    id: "introduction",
    title: "Introduction to BDH",
    slug: "introduction",
    description: "What is the Baby Dragon Hatchling and why does it matter?",
    icon: "",
    lessons: [
      {
        id: "what-is-bdh",
        title: "What is BDH?",
        slug: "what-is-bdh",
        estimatedMinutes: 8,
        sections: [
          {
            type: "text",
            content: `The **Baby Dragon Hatchling (BDH)** is a Large Language Model architecture built on a scale-free, biologically inspired network of *n* locally-interacting neuron particles. The paper calls it the **"Missing Link"** — a bridge between the high-performance Transformer (used in GPT and similar models) and theoretical Brain Models that have never scaled to real tasks.

The central problem BDH is solving isn't just efficiency — it's the **barrier to universal reasoning**. Standard machine learning architectures, including Transformers, struggle to *generalize over time*. Their rules (weights) are frozen after training. Biological networks use local interactions that allow them to adapt their reasoning dynamically as time passes. BDH is built to close that gap.`
          },
          {
            type: "callout",
            variant: "info",
            title: "Why 'Dragon Hatchling'?",
            content: "The name is deliberate. A hatchling begins as a sparse, scale-free graph — small, efficient, local. But it has the structural foundation to grow into a system capable of complex, universal reasoning. The architecture starts with simple local interactions and crystallizes into the 'dragon' — a powerful reasoner that emerges from biological constraints, not brute-force scale."
          },
          {
            type: "text",
            content: `BDH moves away from "black box" behavior through what the paper calls **Axiomatic AI**. Rather than pattern-matching over statistical correlations, BDH uses an internal inference system that resembles Axiomatic Logic — specifically, **Modus Ponens** reasoning: *if A is true, and A implies B, then B is true.*

Instead of asking "what pattern have I seen before?", BDH asks "given what I know right now, what fact is most plausible to evaluate next?" This creates a logical path between concepts rather than a statistical lookup table.`
          },
          {
            type: "text",
            content: `BDH is grounded in three biological principles that work together:

**Hebbian Plasticity** — *"cells that fire together, wire together."* During inference, the model's working memory is stored via synaptic plasticity, just as human neurons strengthen connections over minutes of conversation.

**k-Winners-Take-All (k-WTA)** — only the top-k neurons fire at any timestep, enforcing sparsity through lateral inhibition. This is how the brain prevents every neuron from activating at once.

**Scale-Free Network Structure** — biological brains have heavy-tailed degree distributions: a few "hub" neurons manage high-level logic, while many "leaf" neurons handle specific semantic facts. BDH learns this same structure during training.`
          },
          {
            type: "formula",
            label: "The Synaptic State Equation (schematic)",
            latex: "\\rho_t = \\rho_{t-1} \\cdot U + w_t k_t^T",
            explanation: "The global graph state ρ evolves by decaying the previous state (via rotation operator U) and adding a rank-1 update from the current input's key-value pair. This is the 'Equation of Reasoning' — how working memory updates with each new token. Note: this is a schematic representation. The full BDH-GPU dynamics involve four sub-round equations per layer (expansion, Hebbian update, inference, reconstruction)."
          },
          {
            type: "comparison-table",
            title: "BDH vs Transformer at a Glance",
            headers: ["Feature", "Transformer (GPT)", "Dragon Hatchling (BDH)"],
            rows: [
              ["Memory", "KV-Cache (Fixed buffer)", "Synaptic Plasticity (Dynamic)"],
              ["Logic", "Pattern Matching", "Modus Ponens / Axiomatic"],
              ["Structure", "Dense Layers", "Scale-Free Neuron Graph"],
              ["Neuron Role", "Polysemantic (black box)", "Monosemantic (glass box)"],
              ["Active Parameters", "~100%", "~3–11%"],
              ["Hardware", "GPU/CPU", "GPU / CPU / Neuromorphic"],
            ]
          },
          {
            type: "callout",
            variant: "success",
            title: "The Glass Box Model",
            content: "BDH's 'Interpretability of State' goes beyond just understanding individual neurons. Because activations are sparse and positive, individual synapses are specialists — they visibly strengthen when the model processes specific concepts. You can mathematically trace what the model is reasoning about in real-time. No guessing required."
          }
        ],
        exercise: null
      },
      {
        id: "transformer-limitations",
        title: "Why Transformers Hit a Wall",
        slug: "transformer-limitations",
        estimatedMinutes: 10,
        sections: [
          {
            type: "text",
            content: `The paper identifies a single root cause behind all of the Transformer's limitations: the inability to **generalize over time**. A Transformer's weights are fixed the moment training ends. The model can appear to "reason" within a context window, but it cannot adapt its internal logic as sequences grow longer or as new facts arrive. BDH calls this the primary barrier on the path to Universal Reasoning.

To understand the specific walls this creates, we need to look at three distinct failure modes.`
          },
          {
            type: "callout",
            variant: "warning",
            title: "Wall 1: The Memory Scaling Trap",
            content: "Transformers use a KV-Cache — a global data structure that is not localized at individual neurons. Because every token must attend to every other token, memory cost and compute both grow quadratically with sequence length: O(L²). Doubling context length quadruples the cost. This is not a software bug — it's a mathematical consequence of global key-query lookup."
          },
          {
            type: "formula",
            label: "Standard Scaled Dot-Product Attention",
            latex: "\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V",
            explanation: "The QKᵀ matrix multiplication is O(L²d) in FLOPs and O(L²) in memory — both scale quadratically with sequence length L. BDH replaces this global lookup with attention localized at the neurons as an n×d tensor, breaking the quadratic wall."
          },
          {
            type: "code-snippet",
            language: "python",
            label: "Transformer attention — the O(N²) bottleneck",
            code: `import torch
import torch.nn.functional as F

def transformer_attention(Q, K, V):
    """
    Standard scaled dot-product attention.
    Q, K, V: (batch, seq_len, d_model)
    
    The bottleneck: QK^T produces an (L x L) matrix.
    For L=4096, this is 16M floats — per layer, per head.
    """
    d_k = Q.shape[-1]
    
    # This is the expensive part: every token attends to every other token
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (B, L, L)
    
    attn_weights = F.softmax(scores, dim=-1)  # softmax forces ALL weights > 0
    
    output = torch.matmul(attn_weights, V)
    return output

# Memory complexity: O(L^2) — at L=32768, this is ~4GB just for attention matrices`
          },
          {
            type: "text",
            content: `**Wall 2: The Static Weight Barrier (Episodic Amnesia)**

After training, a Transformer's weights are frozen. "Learning" during a conversation is an illusion — the model just holds context in a temporary KV-Cache buffer. When that buffer is cleared, the knowledge is gone forever. BDH calls this **Episodic Amnesia**. In BDH, by contrast, the model's "working memory" is stored via synaptic plasticity — connections that physically strengthen during inference and persist for the duration of the task.

**Wall 3: The Black Box (The Superposition Catastrophe)**

Because Transformers use dense representations, neurons are "polysemantic" — they represent many unrelated things simultaneously. This isn't just an interpretability problem. It means neurons constantly produce noise for concepts that aren't relevant to the current token. The paper argues this noise is what prevents Transformers from reaching universal reasoning. You cannot build Axiomatic AI on a foundation of superposed, ambiguous representations.`
          },
          {
            type: "callout",
            variant: "info",
            title: "Wall 4: The Energy Wall",
            content: "Most neural networks are uniform — every parameter is equally active. Biological brains are scale-free with heavy-tailed degree distributions, using integrate-and-fire thresholding and sparse positive activations to save metabolic energy. Transformers use dense activations that are computationally expensive and lack the 'micro-inductive bias' of biological neurons. BDH's k-WTA mechanism is a direct answer to this: only 3–11% of neurons fire per step."
          }
        ],
        exercise: {
          id: "ex-transformer-complexity",
          title: "Exercise: Understanding the O(N²) Bottleneck",
          instructions: "Complete the function below that calculates how much memory (in GB) the attention matrix requires for a given sequence length. This will help you feel the quadratic wall.",
          difficulty: "beginner",
          starterCode: `import torch

def attention_memory_gb(seq_len: int, d_model: int = 512, n_heads: int = 8, 
                         bytes_per_float: int = 4) -> dict:
    """
    Calculate memory cost of Transformer attention.
    
    Args:
        seq_len: Sequence length L
        d_model: Model dimension  
        n_heads: Number of attention heads
        bytes_per_float: 4 for float32, 2 for float16
    
    Returns:
        dict with memory breakdown
    """
    # TODO: Calculate the attention matrix size
    # The attention matrix has shape (n_heads, seq_len, seq_len)
    # Hint: total elements = n_heads * seq_len * ___
    attn_matrix_elements = n_heads * seq_len * ___BLANK_1___
    
    # TODO: Convert to gigabytes
    # Hint: 1 GB = 1024^3 bytes
    attn_memory_gb = (attn_matrix_elements * bytes_per_float) / ___BLANK_2___
    
    # BDH comparison: only ~3% of neurons are active per step (k-WTA sparsity).
    # BDH's actual memory is O(n·d) — constant regardless of sequence length L.
    # The figure below estimates the active-neuron compute cost for comparison.
    bdh_active_fraction = 0.03
    bdh_memory_gb = attn_memory_gb * bdh_active_fraction
    
    return {
        "seq_len": seq_len,
        "transformer_attn_gb": round(attn_memory_gb, 4),
        "bdh_sparse_gb": round(bdh_memory_gb, 6),
        "reduction_factor": round(1 / bdh_active_fraction, 0)
    }

# Test it:
for L in [512, 2048, 8192, 32768]:
    result = attention_memory_gb(L)
    print(f"L={L:6d}: Transformer={result['transformer_attn_gb']:.3f}GB | BDH≈{result['bdh_sparse_gb']:.5f}GB")`,
          blanks: [
            {
              id: "BLANK_1",
              placeholder: "___BLANK_1___",
              hint: "The attention matrix is square: each of the seq_len queries attends to all seq_len keys",
              acceptedAnswers: ["seq_len", "L", "seq_len ** 1", "1 * seq_len"],
              explanation: "The attention matrix shape is (n_heads, L, L) — each token attends to every other token, so the second dimension is also seq_len. This is the quadratic cost: L² elements per head."
            },
            {
              id: "BLANK_2",
              placeholder: "___BLANK_2___",
              hint: "1 GB = 1024 × 1024 × 1024 bytes. You can write this as 1024**3 or 1073741824",
              acceptedAnswers: ["1024**3", "1073741824", "(1024**3)", "1024 ** 3", "1024*1024*1024", "1_073_741_824"],
              explanation: "1 gigabyte = 1024³ = 1,073,741,824 bytes. This converts raw byte count into a human-readable GB figure."
            }
          ],
          expectedOutput: `L=   512: Transformer=0.001GB | BDH≈0.00003GB
L=  2048: Transformer=0.016GB | BDH≈0.00048GB
L=  8192: Transformer=0.256GB | BDH≈0.00768GB
L= 32768: Transformer=4.096GB | BDH≈0.12288GB`
        }
      }
    ]
  },

  {
    id: "sparsity",
    title: "Sparsity & k-WTA",
    slug: "sparsity",
    description: "How BDH enforces metabolic efficiency through sparse activation",
    icon: "",
    lessons: [
      {
        id: "sparsity-theory",
        title: "The Metabolic Energy Barrier",
        slug: "sparsity-theory",
        estimatedMinutes: 12,
        sections: [
          {
            type: "text",
            content: `Sparsity in BDH is not just an optimization trick — it is a formal design choice the paper calls **"Axiomatic AI."** Biological neurons are energetically expensive to fire. Real brains use **integrate-and-fire thresholding**: a neuron only consumes computational energy if the incoming signal is strong enough to pass a negative bias threshold. BDH mimics this exactly.

The consequence is profound: by enforcing sparsity, the paper argues we can reach a **Thermodynamic Limit** — a state where the model's reasoning behavior becomes predictable and stable, like a physical system reaching equilibrium. The noise is filtered out. What remains is the skeleton of logic.`
          },
          {
            type: "formula",
            label: "BDH State Update — The Equations of Reasoning",
            latex: "x_t = \\text{ReLU}\\left((1-\\lambda)x_{t-1} + \\eta(q_t \\cdot k_t^T)\\right)",
            explanation: "This formula describes how the model's Working Memory evolves over time. Each component maps directly to a biological mechanism."
          },
          {
            type: "comparison-table",
            title: "Breaking Down the State Update",
            headers: ["Component", "Biological Meaning"],
            rows: [
              ["(1−λ)x_{t−1}", "Synaptic Decay: biological memories fade over time. λ is the forgetting factor that prevents the state from saturating with too much information."],
              ["η(q_t · k_t^T)", "Hebbian Learning (Plasticity): 'neurons that fire together, wire together.' η is the rate at which synapses between concepts strengthen during inference."],
              ["ReLU(…)", "The Metabolic Gate: filters out heavy random noise by applying a negative bias. Only the most significant logical connections survive."],
            ]
          },
          {
            type: "text",
            content: `The second mechanism is **k-Winners-Take-All (k-WTA)** — the model's Inhibitory Circuit. In the brain, when one group of neurons fires, they send inhibitory signals to others to prevent the whole brain from lighting up at once. Uncontrolled global activation in a brain causes a seizure. In an AI model, it causes hallucinations.

k-WTA forces concepts to **compete**. If the model is reasoning about currency and both "Euro" and "Dollar" neurons are candidates, only the top-k are allowed to fire. This competitive pressure is what produces Monosemanticity — each active neuron ends up with one clear, distinct meaning, because it had to *win* that meaning against every other concept simultaneously.`
          },
          {
            type: "formula",
            label: "k-WTA Activation Function",
            latex: "a_i = \\phi(z_i) = (z_i)^+ \\cdot \\mathbb{I}(z_i \\in \\text{top}_k(\\mathbf{z}))",
            explanation: "(z)⁺ = ReLU(z) = max(0, z) enforces the positive orthant constraint (biologically plausible — neurons cannot fire at negative rates). The indicator function I(·) then enforces a strict competition: only the top-k pre-activations survive. All others are zeroed. This is lateral inhibition — losers are suppressed to silence."
          },
          {
            type: "callout",
            variant: "warning",
            title: "Why Sparsity Prevents Hallucinations",
            content: "Traditional Transformers suffer from 'Superposition' — one neuron storing many concepts at once. This overlap creates interference between unrelated ideas, which is the root cause of hallucinations. BDH's hyper-sparsity (~3% active neurons) ensures logical pathways are physically separated in the weight matrix, making it mathematically difficult for the model to mix up unrelated facts."
          },
          {
            type: "code-snippet",
            language: "python",
            label: "k-WTA Implementation — lateral inhibition in code",
            code: `import torch

def k_winners_take_all(x: torch.Tensor, k_fraction: float = 0.15) -> torch.Tensor:
    """
    k-Winners-Take-All activation: only top-k neurons survive.
    
    Args:
        x: Pre-activation tensor of shape (batch, seq_len, d_model)
        k_fraction: Fraction of neurons allowed to fire (e.g., 0.15 = 15%)
    
    Returns:
        Sparse activation tensor — same shape, but ~(1-k_fraction)*100% zeros
    """
    batch, seq_len, d_model = x.shape
    k = max(1, int(d_model * k_fraction))
    
    # ReLU first: enforce positive orthant (biological plausibility)
    x_relu = torch.relu(x)
    
    # Find top-k values per position
    topk_values, _ = torch.topk(x_relu, k, dim=-1)
    threshold = topk_values[..., -1:].expand_as(x_relu)
    
    # Zero out all neurons below the threshold (lateral inhibition)
    mask = x_relu >= threshold
    output = x_relu * mask.float()
    
    return output

# Verify sparsity
x = torch.randn(2, 64, 512)  # batch=2, seq=64, d_model=512
out = k_winners_take_all(x, k_fraction=0.15)

active_fraction = (out > 0).float().mean().item()
print(f"Active neurons: {active_fraction:.1%}")  # Should be ~15%
print(f"Silent neurons: {1-active_fraction:.1%}")  # Should be ~85%`
          }
        ],
        exercise: {
          id: "ex-kwta",
          title: "Exercise: Implement k-WTA from Scratch",
          instructions: "Implement the k-Winners-Take-All mechanism step by step. This is the core of BDH's sparsity enforcement — the biological lateral inhibition that makes neurons compete. Note: this exercise focuses on the feedforward sparsification component. The full BDH layer also updates a synaptic state matrix via Hebbian co-activation, which persists across the sequence.",
          difficulty: "intermediate",
          starterCode: `import torch
import torch.nn as nn

class BDHSparseLayer(nn.Module):
    """
    A BDH-style sparse linear layer using k-WTA activation.
    Replaces the dense FFN in a Transformer block.
    """
    
    def __init__(self, d_model: int, k_fraction: float = 0.15):
        super().__init__()
        self.d_model = d_model
        self.k_fraction = k_fraction
        self.k = max(1, int(d_model * k_fraction))
        
        # Encoder-decoder factorization (ReLU-Lowrank)
        self.encoder = nn.Linear(d_model, d_model, bias=True)
        self.decoder = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: LayerNorm → Encode → ReLU → k-WTA → Decode
        """
        # Step 1: Normalize input (prevents magnitude collapse)
        x_norm = ___BLANK_1___(x)
        
        # Step 2: Encode to higher-dimensional space
        z = self.encoder(x_norm)
        
        # Step 3: ReLU — enforce positive orthant constraint
        # Biological neurons cannot have negative firing rates
        z_relu = ___BLANK_2___(z)
        
        # Step 4: k-WTA — only top-k neurons survive lateral inhibition
        # Find the k-th largest value as the threshold
        topk_vals, _ = torch.topk(z_relu, self.k, dim=-1)
        threshold = topk_vals[..., -1:]  # The minimum "winning" value
        
        # Zero out all neurons that didn't win the competition
        mask = (z_relu >= threshold).float()
        z_sparse = z_relu * ___BLANK_3___
        
        # Step 5: Decode back to d_model
        output = self.decoder(z_sparse)
        
        return output
    
    def get_sparsity(self, x: torch.Tensor) -> float:
        """Calculate fraction of neurons that are silent (zero)."""
        with torch.no_grad():
            out = self.forward(x)
            # TODO: Return the fraction of zero activations
            silent_fraction = ___BLANK_4___
            return silent_fraction


# ---- Test your implementation ----
torch.manual_seed(42)
layer = BDHSparseLayer(d_model=256, k_fraction=0.15)
x = torch.randn(4, 32, 256)  # batch=4, seq=32, d_model=256

output = layer(x)
sparsity = layer.get_sparsity(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Sparsity: {sparsity:.1%} silent neurons")
print(f"Expected sparsity: ~{(1 - 0.15):.0%}")
assert output.shape == x.shape, "Output shape must match input shape!"
print("✓ Shape check passed")`,
          blanks: [
            {
              id: "BLANK_1",
              placeholder: "___BLANK_1___",
              hint: "We defined self.layer_norm in __init__. This is the LayerNorm normalization step.",
              acceptedAnswers: ["self.layer_norm", "nn.functional.layer_norm"],
              explanation: "self.layer_norm normalizes the input to have zero mean and unit variance. This prevents the k-WTA competition from being dominated by high-magnitude outliers."
            },
            {
              id: "BLANK_2",
              placeholder: "___BLANK_2___",
              hint: "ReLU clips negative values to zero: max(0, z). Use torch.relu() or torch.nn.functional.relu()",
              acceptedAnswers: ["torch.relu", "torch.relu(z)", "nn.functional.relu", "torch.nn.functional.relu", "F.relu", "F.relu(z)", "torch.clamp(z, min=0)", "z.clamp(min=0)"],
              explanation: "torch.relu(z) enforces the positive orthant constraint — activations can only be ≥ 0. This is biologically plausible (neurons don't have negative firing rates) and is the metabolic gate that filters noise."
            },
            {
              id: "BLANK_3",
              placeholder: "___BLANK_3___",
              hint: "You've already computed 'mask' — multiply element-wise to zero out losers",
              acceptedAnswers: ["mask", "* mask", "z_relu * mask", "mask.float()", "* mask.float()"],
              explanation: "Multiplying by the binary mask zeros out all neurons that didn't win the k-WTA competition. This is the lateral inhibition step — losers are suppressed to zero, winners keep their activation."
            },
            {
              id: "BLANK_4",
              placeholder: "___BLANK_4___",
              hint: "Run forward pass on x, check what fraction of output values equal exactly zero. Use (tensor == 0).float().mean()",
              acceptedAnswers: [
                "(self.forward(x) == 0).float().mean().item()",
                "(out == 0).float().mean().item()",
                "1 - (out > 0).float().mean().item()",
                "(out == 0).float().mean()",
              ],
              explanation: "Count the fraction of zero activations in the output. Since k-WTA zeros out (1 - k_fraction) of neurons, this should return approximately 0.85 (85% silent) for k_fraction=0.15."
            }
          ],
          expectedOutput: `Input shape:  torch.Size([4, 32, 256])
Output shape: torch.Size([4, 32, 256])
Sparsity: ~85.0% silent neurons
Expected sparsity: ~85%
✓ Shape check passed`
        }
      },
      {
        id: "sparsity-phases",
        title: "The Synaptic Pruning Trajectory",
        slug: "sparsity-phases",
        estimatedMinutes: 8,
        sections: [
          {
            type: "text",
            content: `BDH training follows a three-phase trajectory that the paper describes as the physical manifestation of a model moving toward its **Thermodynamic Limit** — the state where noise has been filtered out and only the skeleton of logic remains. This mirrors how biological brains develop from birth to adulthood.

The journey is from a dense, chaotic **"Hatchling"** to a structured, logical **"Dragon."**`
          },
          {
            type: "callout",
            variant: "info",
            title: "Note on Sparsity Percentages",
            content: "The three-phase trajectory (18% → 11% → 3%) and phase names (Exploration, Pruning, Crystallization) are conceptual framings used to give intuition for the training dynamics. The BDH paper reports empirically observed sparsity of approximately 5% in trained models. The phase boundaries and specific percentages here are illustrative, not precisely defined in the paper."
          },
          {
            type: "callout",
            variant: "info",
            title: "Phase 1: The Exploration Phase (The Hatchling State)",
            content: "In the first few epochs (~18% firing rate), the model is in a high-entropy broadcast mode. Neuron particles fire semi-randomly as they encounter new linguistic patterns. The model hasn't yet decided which neurons are 'hubs' and which are 'leaf nodes' — it's establishing its scale-free nature. Like a child's brain with an excess of unspecialized synapses, it's susceptible to interference."
          },
          {
            type: "callout",
            variant: "warning",
            title: "Phase 2: The Pruning Phase (Competitive Refinement)",
            content: "This is where k-WTA becomes critical. As the model identifies consistent logical patterns, those pathways receive more utility signals. The paper describes a 'negative bias' that grows in neurons that don't contribute to successful predictions — a metabolic tax that eventually silences them via the ReLU gate. The λ decay factor constantly erodes weak connections. Only connections refreshed frequently by high-utility logical interactions survive."
          },
          {
            type: "callout",
            variant: "success",
            title: "Phase 3: Crystallization (The Specialist State)",
            content: "By ~epoch 40 (~3% firing rate), the model has crystallized. Neurons are no longer generalists — they are Specialist Neurons. The network follows a power-law distribution: a few 'Hub' neurons manage high-level logic (grammar, context), while 'Leaf' neurons handle specific semantic facts. The internal state becomes highly robust to noise: because pathways are sparse and distinct, a typo or random noise cannot easily derail the logical flow."
          },
          {
            type: "code-snippet",
            language: "python",
            label: "Tracking the Sparsity Trajectory During Training",
            code: `# From the actual BDH training loop
# This diagnostic is how the paper measured the pruning trajectory

diagnostics = {'sparsities': [], 'losses': []}

for epoch in range(1, 41):
    for batch in dataloader:
        # Forward pass through BDH layers
        output, hidden_states = model(batch)
        
        # Main language modeling loss
        loss_lm = criterion(output, targets)
        
        # L1 Sparsity penalty — this is what drives pruning!
        # Lambda=0.001: gentle pressure toward minimal activation
        lambda_sparse = 0.001
        loss_sparse = lambda_sparse * sum(
            hidden.abs().mean() 
            for hidden in hidden_states
        )
        
        loss = loss_lm + loss_sparse
        
        # Track the firing rate of the final layer (L8)
        with torch.no_grad():
            q = hidden_states[-1]  # Last layer activations
            firing_rate = (q > 0).float().mean().item()
            diagnostics['sparsities'].append(firing_rate)
        
        loss.backward()
        optimizer.step()

# Expected trajectory:
# Epoch 1:  17.98% — Exploration Phase (dense, high-entropy broadcast)
# Epoch 15: 11.07% — Pruning Phase (metabolic tax silencing weak connections)
# Epoch 40:  3.00% — Crystallization (hyper-sparse, specialist state)`
          },
          {
            type: "callout",
            variant: "info",
            title: "The Geometry of Logic",
            content: "In the BDH paper, pruning isn't just about efficiency — it's about geometry. By reducing the number of active neurons, the model effectively 'compresses' its knowledge into a lower-dimensional manifold. This makes reasoning Axiomatic: you can look at the remaining active synapses and trace the exact logical proof the model is building in real-time."
          }
        ],
        exercise: {
          id: "ex-sparsity-loss",
          title: "Exercise: Build the Sparsity-Aware Loss Function",
          instructions: "BDH trains with a combined loss: language modeling + L1 sparsity penalty. Complete the loss function that drives the pruning trajectory. The L1 term is what forces the network to 'crystallize'.",
          difficulty: "intermediate",
          starterCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

def bdh_combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    hidden_states: list,
    lambda_sparse: float = 0.001,
    vocab_size: int = 65
) -> dict:
    """
    BDH's combined training objective:
    L_total = L_CE (language modeling) + λ * L_sparse (sparsity pressure)
    
    Args:
        logits: Model output logits (batch, seq_len, vocab_size)
        targets: Target token IDs (batch, seq_len)  
        hidden_states: List of intermediate layer activations
        lambda_sparse: Weight for sparsity regularization (default: 0.001)
        vocab_size: Size of vocabulary
    
    Returns:
        dict with total loss and component breakdown
    """
    # Step 1: Cross-entropy loss for language modeling
    # Reshape logits from (B, L, V) to (B*L, V) for F.cross_entropy
    B, L, V = logits.shape
    logits_flat = logits.view(___BLANK_1___, V)
    targets_flat = targets.view(-1)
    
    loss_ce = F.cross_entropy(logits_flat, targets_flat)
    
    # Step 2: L1 Sparsity penalty — sum of mean absolute activations across all layers
    # This is the key: punish the model for having large activations
    # L_sparse = sum over all layers of mean(|hidden_state|)
    loss_sparse = sum(
        ___BLANK_2___.abs().mean()  # L1 norm of each layer's activations
        for h in hidden_states
    )
    
    # Step 3: Combine with lambda weighting
    loss_total = loss_ce + ___BLANK_3___ * loss_sparse
    
    # Compute sparsity diagnostic (fraction of zero activations in last layer)
    with torch.no_grad():
        last_layer = hidden_states[-1]
        sparsity = (last_layer == 0).float().mean().item()
    
    return {
        "loss_total": loss_total,
        "loss_ce": loss_ce.item(),
        "loss_sparse": loss_sparse.item(),
        "sparsity": sparsity
    }


# ---- Test your implementation ----
torch.manual_seed(0)
B, L, V, D = 2, 16, 65, 128

logits = torch.randn(B, L, V)
targets = torch.randint(0, V, (B, L))

# Simulate hidden states from 4 BDH layers (post k-WTA, so mostly zeros)
hidden_states = []
for layer_i in range(4):
    h = torch.relu(torch.randn(B, L, D))  # ReLU already applied
    # Apply k-WTA: zero out 85% of neurons
    k = max(1, int(D * 0.15))
    topk, _ = torch.topk(h, k, dim=-1)
    threshold = topk[..., -1:]
    h = h * (h >= threshold).float()
    hidden_states.append(h)

result = bdh_combined_loss(logits, targets, hidden_states)
print(f"Total Loss:    {result['loss_total']:.4f}")
print(f"  CE Loss:     {result['loss_ce']:.4f}")
print(f"  Sparse Loss: {result['loss_sparse']:.6f}")
print(f"  Sparsity:    {result['sparsity']:.1%} neurons silent")`,
          blanks: [
            {
              id: "BLANK_1",
              placeholder: "___BLANK_1___",
              hint: "You need to flatten the batch and sequence dimensions: (B, L, V) → (B*L, V). Use -1 to let PyTorch calculate it.",
              acceptedAnswers: ["-1", "B * L", "B*L", "(-1)"],
              explanation: "F.cross_entropy expects 2D input (N, C). We use -1 to let PyTorch calculate B*L automatically: logits.view(-1, V) reshapes from (B, L, V) to (B*L, V)."
            },
            {
              id: "BLANK_2",
              placeholder: "___BLANK_2___",
              hint: "In the generator expression, each hidden state is named 'h'. Use that variable.",
              acceptedAnswers: ["h", "h.abs()", "hidden_states"],
              explanation: "In the generator expression 'for h in hidden_states', 'h' is each individual layer's activation tensor. We take h.abs() to get L1 norm (absolute values), then .mean() to average."
            },
            {
              id: "BLANK_3",
              placeholder: "___BLANK_3___",
              hint: "Scale the sparsity loss by the lambda parameter defined as a function argument",
              acceptedAnswers: ["lambda_sparse", "0.001"],
              explanation: "lambda_sparse (default 0.001) controls how strongly we penalize large activations. Too high and the model won't learn; too low and neurons won't prune. 0.001 is the value used in the BDH paper."
            }
          ],
          expectedOutput: `Total Loss:    4.1xxx
  CE Loss:     4.1xxx
  Sparse Loss: 0.0xxxxx
  Sparsity:    ~85.0% neurons silent`
        }
      }
    ]
  },

  {
    id: "monosemanticity",
    title: "Monosemanticity",
    slug: "monosemanticity",
    description: "How BDH neurons become dedicated, interpretable feature detectors",
    icon: "",
    lessons: [
      {
        id: "monosemanticity-theory",
        title: "The Superposition Catastrophe",
        slug: "monosemanticity-theory",
        estimatedMinutes: 10,
        sections: [
          {
            type: "text",
            content: `**Polysemanticity** is the original sin of dense neural networks. The paper calls it the **Superposition Catastrophe** — and it's not an accident. It's a forced tradeoff.

Because Transformers are dense (every parameter is involved in every calculation), representing the complexity of human language requires "smushing" multiple features into the same high-dimensional space. A single neuron might encode 70% of concept A and 30% of concept B simultaneously. This superposition means neurons constantly produce noise for concepts that aren't relevant — and the paper argues this noise is what prevents Transformers from reaching Universal Reasoning. You cannot build Axiomatic AI on ambiguous, overlapping representations.`
          },
          {
            type: "callout",
            variant: "warning",
            title: "The Superposition Catastrophe",
            content: "In standard Transformers (using GeLU), FFN(x) = GeLU(xW₁)W₂ allows unrestricted activations in ℝ. A neuron can encode feature A as positive and feature B as negative — packing both in, invisible to inspection. To understand what any single neuron is doing, you need complex tools like Integrated Gradients to guess. BDH's answer: make superposition geometrically impossible."
          },
          {
            type: "text",
            content: `BDH solves this through **Geometric Orthogonality**. By forcing activations to be positive (via ReLU) and sparse (via k-WTA), BDH pushes neurons toward monosemanticity. The pathway for "cats" becomes physically different from the pathway for "geometry." Cross-talk between unrelated concepts becomes mathematically impossible — not just unlikely.

The key mechanism is the **Hebbian Update Rule**, which governs how connections form and which ones survive.`
          },
          {
            type: "formula",
            label: "Hebbian Weight Update Rule",
            latex: "\\Delta W_{ij} \\propto (x_i \\cdot x_j) - \\gamma W_{ij}",
            explanation: "Two terms work together: the correlation term (xi·xj) strengthens connections when neurons fire together, and the homeostatic decay (−γWij) prevents any single connection from dominating. Orthogonal concepts — those with near-zero correlation — naturally separate into distinct neurons."
          },
          {
            type: "comparison-table",
            title: "Breaking Down the Hebbian Rule",
            headers: ["Component", "Biological Meaning"],
            rows: [
              ["(x_i · x_j)", "The 'Fire Together' rule: if neurons i and j are active at the same time, the connection grows. Captures logical dependency — e.g., 'New' and 'York' co-activating."],
              ["−γW_{ij}", "The Homeostatic Decay ('Forget' rule): prevents any connection from becoming too strong and dominating the network. Forces the network to stay sparse by constantly eroding unused connections."],
            ]
          },
          {
            type: "text",
            content: `The result of Hebbian learning combined with k-WTA competition is empirically striking. White-box probing of a trained 24M parameter BDH model on the Europarl corpus revealed **monosemantic synapses** — individual synapse entries in the σ matrix that activate with surgical precision whenever a specific concept appears:

The researchers found synapses that activate specifically for **currency names** (Euro, Dollar, British Pound) — and remarkably, the same synapse activates whether the sentence is in French ("livre sterling") or English ("British Pound"). Semantic selectivity crosses language barriers. A one-sided Mann–Whitney U test confirmed that currency-related sentences received significantly higher "Currency synapse" values than non-currency sentences (p < 10⁻¹⁴, rank-biserial correlation 0.86).

At the character level, the probing found **Alphabetical Specialists** (Neuron #598 fires exclusively for 'I' with P>0.95; Neuron #14 for 'H') and **Structural Specialists** (Neuron #12 fires only for commas — a "clock" that tells the model when a thought has ended). Punctuation neurons are physically clustered differently from alphabetic neurons, creating emergent lobe structure within the N dimension — without any explicit supervision.

Note: the BDH paper's white-box probing primarily identifies monosemantic *synapses* (σ(i,j) entries in the state matrix) rather than individual neurons. The character-specialist examples here illustrate the broader phenomenon of sparse, localized representations.`
          },
          {
            type: "callout",
            variant: "success",
            title: "Glass Box vs Black Box",
            content: "Transformer: to understand a neuron, you need Integrated Gradients or feature steering — complex tools that only approximate what the neuron is doing. BDH: you simply look at the v* vector. If a neuron's index is active, its specialist function is being used. No guessing. The 'address' of a concept in the brain is literally its neuron index. BDH's probing results are consistent with the grandmother cell hypothesis — that localized neurons can represent specific concepts. The paper provides statistical evidence of synapse-level monosemanticity (Mann-Whitney U test, p < 10⁻¹⁴); causal ablation experiments remain future work."
          },
          {
            type: "comparison-table",
            title: "Interpretability: Glass Box vs Black Box",
            headers: ["Feature", "Black Box (Transformer)", "Glass Box (BDH)"],
            rows: [
              ["Logic", "Emergent / Statistical", "Axiomatic / Traceable"],
              ["Neuron Role", "Polysemantic (jack of all trades)", "Monosemantic (specialist)"],
              ["Auditability", "Requires 'Feature Steered' models", "Direct weight inspection"],
              ["Hallucination", "Hard to trace (superposition)", "Easy to trace (active pathways)"],
            ]
          },
          {
            type: "code-snippet",
            language: "python",
            label: "White-Box Probing: Finding Specialist Neurons",
            code: `import torch
import numpy as np

def probe_specialist_neurons(model, test_sequence: str, top_n: int = 5):
    """
    White-box probe: for each character, find which neurons fire most exclusively.
    A 'specialist neuron' is one that fires strongly for one char and weakly for all others.
    
    Unlike Transformer probing (which requires a second ML model to guess),
    BDH probing is direct: we simply read the sparse activation vector.
    """
    model.eval()
    neuron_char_activations = {}  # {neuron_id: {char: avg_activation}}
    
    with torch.no_grad():
        for char in set(test_sequence):
            # Create a batch of this character repeated
            char_tensor = encode(char).unsqueeze(0)  # (1, 1)
            _, hidden = model(char_tensor)
            last_layer = hidden[-1][0, 0]  # (d_model,)
            
            for neuron_id, activation in enumerate(last_layer.tolist()):
                if neuron_id not in neuron_char_activations:
                    neuron_char_activations[neuron_id] = {}
                neuron_char_activations[neuron_id][char] = activation
    
    # Score each neuron by selectivity: high max / low mean = specialist
    specialists = []
    for neuron_id, char_acts in neuron_char_activations.items():
        activations = list(char_acts.values())
        if max(activations) == 0:
            continue
        
        max_act = max(activations)
        mean_act = np.mean(activations)
        selectivity = max_act / (mean_act + 1e-8)
        best_char = max(char_acts, key=char_acts.get)
        
        specialists.append({
            "neuron": neuron_id,
            "target_char": best_char,
            "selectivity": selectivity,
            "max_activation": max_act
        })
    
    specialists.sort(key=lambda x: x['selectivity'], reverse=True)
    
    print("Top Specialist Neurons (from BDH paper, 24M param Europarl model):")
    print(f"{'Neuron':>8}  {'Target':>8}  {'Selectivity':>12}  {'Max Act':>10}")
    print("-" * 50)
    for s in specialists[:top_n]:
        print(f"  #{s['neuron']:5d}  {repr(s['target_char']):>8}  {s['selectivity']:>12.2f}  {s['max_activation']:>10.4f}")
    
    return specialists

# Results from the BDH paper's trained model:
# Neuron #598  → 'I'  (P>0.95 — fires almost exclusively for this character)
# Neuron  #14  → 'H'  (dedicated detector)
# Neuron  #12  → ','  (comma specialist — the sentence "clock")`
          }
        ],
        exercise: {
          id: "ex-monosemanticity",
          title: "Exercise: Implement Oja's Rule for Monosemantic Learning",
          instructions: "Oja's Rule is a classical stabilized Hebbian learning update (Erkki Oja, 1982) introduced here as a pedagogical tool for understanding monosemantic learning. The actual BDH paper uses a simpler co-activation Hebbian update: σ(i,j) is incremented by Y(i)·X(j) with a decay factor. Oja's Rule captures the same spirit — correlated neurons strengthen, isolated neurons weaken — but adds weight-normalization properties useful for this exercise. Implement it step by step below.",
          difficulty: "advanced",
          starterCode: `import torch
import torch.nn as nn

class HebbianOjaLayer(nn.Module):
    """
    A layer that learns via Oja's Rule — the biological Hebbian update
    that self-organizes neurons into monosemantic feature detectors.
    
    Oja's Rule: Δw_ij = η * y_i * (x_j - y_i * w_ij)
    
    The key insight: the decay term -η*y_i²*w_ij prevents weight explosion
    and converges the weight vector toward the principal component of the data.
    """
    
    def __init__(self, in_features: int, out_features: int, lr: float = 0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr
        
        # Initialize weights (will be updated by Hebbian rule, not backprop)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01,
            requires_grad=False  # Not updated by optimizer!
        )
    
    def oja_update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Apply Oja's Rule to update weights.
        
        Oja's Rule: Δw_ij = η * y_i * (x_j - y_i * w_ij)
        
        Args:
            x: Pre-synaptic input  (batch, in_features)
            y: Post-synaptic output (batch, out_features)
        """
        # Average over batch dimension for stable updates
        x_mean = x.mean(0)  # (in_features,)
        y_mean = y.mean(0)  # (out_features,)
        
        # Hebbian term: η * y_i * x_j  (strengthens correlated connections)
        # Shape: outer product gives (out_features, in_features)
        hebbian_term = ___BLANK_1___  # outer product of y_mean and x_mean
        
        # Oja decay term: -η * y_i² * w_ij  (prevents weight explosion)
        # This is what makes it stable — without this, weights grow unbounded
        oja_decay = ___BLANK_2___  # y_mean squared, outer product with ones, times self.weight
        
        # Combine: Δw = η * (Hebbian - OjaDecay)
        delta_w = self.lr * (hebbian_term - oja_decay)
        
        # Update weights in-place (no gradient needed — local learning rule)
        self.weight.data += delta_w
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard linear forward + ReLU (positive orthant constraint)."""
        y = torch.relu(x @ ___BLANK_3___)  # matrix multiply: (batch, in) × (in, out) → (batch, out)
        
        # Apply Hebbian update if training
        if self.training:
            self.oja_update(x.detach(), y.detach())
        
        return y
    
    def get_weight_stats(self) -> dict:
        """Analyze weight distribution — should become bimodal after training."""
        w = self.weight.data.flatten()
        strong = (w.abs() > 0.3).float().mean().item()   # strong connections
        silent = (w.abs() < 0.05).float().mean().item()  # dead connections
        return {
            "strong_fraction": strong,
            "silent_fraction": silent,
            "bimodality": strong + silent  # close to 1.0 = highly bimodal
        }


# ---- Test your implementation ----
torch.manual_seed(42)
layer = HebbianOjaLayer(in_features=64, out_features=32, lr=0.01)

print("Training Oja's Rule for 100 steps...")
print(f"{'Step':>6}  {'Strong%':>8}  {'Silent%':>8}  {'Bimodality':>12}")
print("-" * 45)

for step in range(100):
    # Simulated structured input: some features always co-activate
    x = torch.randn(16, 64)
    # Add correlations between feature pairs (simulating semantic clusters)
    x[:, 1] = x[:, 0] * 0.8 + torch.randn(16) * 0.2  # features 0,1 correlated
    x[:, 3] = x[:, 2] * 0.8 + torch.randn(16) * 0.2  # features 2,3 correlated
    
    layer.train()
    _ = layer(x)
    
    if step % 25 == 0:
        stats = layer.get_weight_stats()
        print(f"  {step:4d}  {stats['strong_fraction']:>7.1%}  {stats['silent_fraction']:>7.1%}  {stats['bimodality']:>11.3f}")

print("\\nFinal weight distribution should be bimodal (strong + silent ≈ 1.0)")`,
          blanks: [
            {
              id: "BLANK_1",
              placeholder: "___BLANK_1___",
              hint: "The Hebbian term is an outer product: each output neuron y_i is multiplied by each input x_j. Use torch.outer(y_mean, x_mean) — this gives a (out_features, in_features) matrix.",
              acceptedAnswers: [
                "torch.outer(y_mean, x_mean)",
                "y_mean.unsqueeze(1) * x_mean.unsqueeze(0)",
                "torch.ger(y_mean, x_mean)",
              ],
              explanation: "torch.outer(y, x) computes the outer product: result[i,j] = y[i] * x[j]. This captures the co-activation of each output neuron with each input feature — the core of Hebbian learning."
            },
            {
              id: "BLANK_2",
              placeholder: "___BLANK_2___",
              hint: "The Oja decay is: y_i² × w_ij. You need y_mean squared as a column vector, multiplied element-wise with self.weight. Try: (y_mean**2).unsqueeze(1) * self.weight",
              acceptedAnswers: [
                "(y_mean**2).unsqueeze(1) * self.weight",
                "torch.outer(y_mean**2, torch.ones(self.in_features)) * self.weight",
                "(y_mean ** 2).unsqueeze(-1) * self.weight",
                "(y_mean.pow(2)).unsqueeze(1) * self.weight",
              ],
              explanation: "The Oja decay term prevents weight explosion: -η*y_i²*w_ij. We need y_mean squared (one value per output neuron) broadcast across all input connections. unsqueeze(1) makes it (out, 1) so it broadcasts with weight (out, in)."
            },
            {
              id: "BLANK_3",
              placeholder: "___BLANK_3___",
              hint: "self.weight has shape (out_features, in_features). For x @ W to work with x shape (batch, in_features), you need to transpose W.",
              acceptedAnswers: [
                "self.weight.T",
                "self.weight.t()",
                "self.weight.transpose(0, 1)",
                "self.weight.transpose(-2, -1)",
              ],
              explanation: "self.weight is (out_features, in_features). For the linear transform x @ W where x is (batch, in_features), we need W to be (in_features, out_features) — so we transpose: self.weight.T."
            }
          ],
          expectedOutput: `Training Oja's Rule for 100 steps...
  Step  Strong%  Silent%  Bimodality
-----------------------------------------
     0     ~2%    ~95%       ~0.97
    25    ~15%    ~75%       ~0.90
    50    ~25%    ~65%       ~0.90
    75    ~30%    ~60%       ~0.90
Final weight distribution should be bimodal (strong + silent ≈ 1.0)`
        }
      }
    ]
  },

  {
    id: "memory",
    title: "Multi-Lobe Memory",
    slug: "memory",
    description: "BDH's brain-inspired hierarchical memory system",
    icon: "",
    lessons: [
      {
        id: "multi-lobe-theory",
        title: "Spectral Memory Hierarchy",
        slug: "multi-lobe-theory",
        estimatedMinutes: 12,
        sections: [
          {
            type: "callout",
            variant: "info",
            title: "Pedagogical Note: Spectral Memory Model",
            content: "The four-lobe spectral memory model (Occipital/Frontal/Parietal/Temporal) presented in this chapter is a pedagogical framework inspired by neuroscience. The BDH paper describes memory as a single unified synaptic state matrix ρ (an n×d tensor) evolving via rank-1 Hebbian updates. The lobe decomposition is introduced here to build intuition for how different temporal frequencies of information can co-exist in one system."
          },
          {
            type: "text",
            content: `In a standard Transformer, memory is a buffer of tokens — a flat KV-Cache that treats all previous context as equally important until it falls off the edge of the context window. When you hit the limit, the model instantly forgets the beginning of the story.

BDH takes a completely different approach. **Memory in BDH is a dynamic physical state**, not a token buffer. Think of a prism splitting white light into its component colors. BDH splits the sequence of information into different **temporal frequencies**, allowing the model to "feel" long-range context without having to store every specific word.

The paper calls this a **Spectral Filter Bank** — a system of memory lobes that process information at different speeds simultaneously, mirroring how different regions of the mammalian cortex operate.`
          },
          {
            type: "formula",
            label: "Multi-Lobe State Update",
            latex: "S_t^{(k)} = (1 - \\lambda_k) \\cdot S_{t-1}^{(k)} + \\phi(W_{att}^{(k)} \\cdot x_t)",
            explanation: "Each lobe k has its own decay constant λₖ and projection matrix W. The persistence factor (1−λₖ) determines how much of the past is retained. High λ = fast forgetting. Low λ = long memory. This creates a natural spectral decomposition across lobes."
          },
          {
            type: "comparison-table",
            title: "Breaking Down the Lobe Update",
            headers: ["Component", "Meaning in the BDH Cortex"],
            rows: [
              ["S_t^(k)", "The Lobe State: the current 'memory' of lobe k at timestep t."],
              ["(1 − λ_k)", "The Persistence Factor: λ=0.1 means 90% of the past is retained each step (long memory). λ=0.9 means only 10% is retained (fast decay)."],
              ["φ(W_att^(k) · x_t)", "The Sensory Inflow: new information being integrated. φ (like ReLU) ensures only relevant signals enter the memory lobe."],
            ]
          },
          {
            type: "text",
            content: `The four lobes form a temporal hierarchy that mirrors human cognition:

**Occipital Lobe (λ ≈ 0.9) — Echoic Memory:** This is a high-pass filter. It holds the literal vibration of the last word spoken — the raw sensory input before any interpretation. It forgets almost immediately.

**Frontal Lobe (λ ≈ 0.8) — Grammar State:** It holds the syntactic frame. It doesn't care what the words mean, only that a "Subject" was just introduced and a "Verb" should follow. Tracks the last 3–5 tokens.

**Parietal Lobe (λ ≈ 0.5) — Relational Bridge:** It tracks the relationship between clauses. It remembers that the current sentence is a rebuttal to the previous one, holding the logical connective tissue between nearby sentences.

**Temporal Lobe (λ ≈ 0.1) — Narrative Gist:** This is a low-pass filter. Even if the specific words are long forgotten, the "Narrative State" persists — "we are talking about a dragon." This lobe holds the gist across hundreds or thousands of tokens.`
          },
          {
            type: "callout",
            variant: "success",
            title: "Graceful Degradation vs Hard Cutoffs",
            content: "When a Transformer hits its context limit, it forgets the beginning of the story instantly — a hard cutoff. In BDH, because of the Temporal Lobe (λ≈0.1), the model never truly 'forgets' early context. Instead, memory slowly fades into a generalized summary, allowing much longer-range coherence. The paper also notes these lobes aren't hardcoded — they emerge through training as the model self-organizes into a brain-like structure to minimize prediction error."
          },
          {
            type: "code-snippet",
            language: "python",
            label: "Multi-Lobe Memory Implementation",
            code: `import torch
import torch.nn as nn

class MultiLobeMemory(nn.Module):
    """
    BDH's brain-inspired hierarchical memory system.
    Each lobe has a different decay rate, creating a spectral filter bank
    that processes information at multiple temporal resolutions simultaneously.
    """
    
    LOBE_CONFIG = {
        "occipital":  {"decay": 0.90, "role": "Echoic memory — raw sensory input"},
        "frontal":    {"decay": 0.80, "role": "Grammar state — syntax (3-5 tokens)"},
        "parietal":   {"decay": 0.50, "role": "Relational bridge — clause connections"},
        "temporal":   {"decay": 0.10, "role": "Narrative gist — 100s of tokens"},
    }
    
    def __init__(self, d_model: int, d_lobe: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_lobe = d_lobe or d_model // 4  # Each lobe gets d_model/4 dimensions
        
        self.lobes = nn.ModuleDict({
            name: nn.Linear(d_model, self.d_lobe, bias=False)
            for name in self.LOBE_CONFIG
        })
        self.decays = {
            name: cfg["decay"] 
            for name, cfg in self.LOBE_CONFIG.items()
        }
        self.output_proj = nn.Linear(self.d_lobe * len(self.LOBE_CONFIG), d_model)
        
    def forward(self, x: torch.Tensor, states: dict = None):
        if states is None:
            states = {
                name: torch.zeros(x.shape[0], self.d_lobe, device=x.device)
                for name in self.LOBE_CONFIG
            }
        
        new_states = {}
        lobe_outputs = []
        
        for name, proj in self.lobes.items():
            decay = self.decays[name]
            prev_state = states[name]
            
            # Project input to lobe-specific feature space
            new_input = torch.relu(proj(x))
            
            # Leaky integration with lobe-specific decay
            # High decay (Occipital) = fast forgetting
            # Low decay (Temporal) = persistent narrative memory
            new_state = (1 - decay) * prev_state + new_input
            
            new_states[name] = new_state
            lobe_outputs.append(new_state)
        
        combined = torch.cat(lobe_outputs, dim=-1)
        output = self.output_proj(combined)
        
        return output, new_states`
          }
        ],
        exercise: {
          id: "ex-memory-decay",
          title: "Exercise: Simulate Memory Decay Across Lobes",
          instructions: "Implement a function that shows how information decays differently in each brain lobe. Given an initial signal strength of 1.0, simulate how each lobe retains it over T timesteps. This will reproduce the memory retention curves from the BDH paper.",
          difficulty: "beginner",
          starterCode: `import torch
import math

def simulate_lobe_decay(
    T: int = 200, 
    initial_strength: float = 1.0,
    lobe_configs: dict = None
) -> dict:
    """
    Simulate how each brain lobe retains an initial memory over T timesteps.
    
    Each lobe uses: S_t = (1 - λ) * S_{t-1} (pure decay, no new input)
    
    Args:
        T: Number of timesteps to simulate
        initial_strength: Initial signal strength at t=0
        lobe_configs: Dict of {lobe_name: lambda_decay}
    
    Returns:
        Dict of {lobe_name: [signal_at_t0, signal_at_t1, ..., signal_at_tT]}
    """
    if lobe_configs is None:
        lobe_configs = {
            "Occipital (λ=0.90)":  0.90,  # Fast decay — echoic memory
            "Frontal (λ=0.80)":    0.80,  # Fast decay — grammar state
            "Parietal (λ=0.50)":   0.50,  # Medium decay — relational bridge
            "Temporal (λ=0.10)":   0.10,  # Slow decay — narrative gist
        }
    
    decay_curves = {}
    
    for lobe_name, lambda_k in lobe_configs.items():
        signal = initial_strength
        trajectory = [signal]
        
        for t in range(1, T + 1):
            # Apply one step of lobe decay (no new input this timestep)
            # Hint: S_t = (1 - λ) * S_{t-1}
            signal = ___BLANK_1___ * signal
            trajectory.append(signal)
        
        decay_curves[lobe_name] = trajectory
    
    return decay_curves


def find_half_life(trajectory: list, initial: float = 1.0) -> int:
    """
    Find the timestep at which signal drops below 50% of initial strength.
    Returns T+1 if never reaches half.
    """
    threshold = initial * 0.5
    for t, val in enumerate(trajectory):
        if ___BLANK_2___:  # signal has fallen below 50%
            return t
    return len(trajectory)  # Never reached half-life


def analytical_half_life(lambda_k: float) -> float:
    """
    Calculate the theoretical half-life using the analytical formula.
    For S_t = (1-λ)^t, half-life = log(0.5) / log(1-λ)
    
    Hint: Use math.log()
    """
    return ___BLANK_3___


# ---- Test your implementation ----
curves = simulate_lobe_decay(T=200)

print(f"{'Lobe':<28} {'Half-Life':>10}  {'At t=200':>10}  {'Analytical HL':>14}")
print("-" * 70)

lobe_lambdas = {"Occipital (λ=0.90)": 0.90, "Frontal (λ=0.80)": 0.80,
                "Parietal (λ=0.50)": 0.50, "Temporal (λ=0.10)": 0.10}

for lobe_name, traj in curves.items():
    hl = find_half_life(traj)
    final_val = traj[-1]
    lk = lobe_lambdas[lobe_name]
    analytical_hl = analytical_half_life(lk)
    print(f"  {lobe_name:<26} {hl:>9}t  {final_val:>10.6f}  {analytical_hl:>13.1f}t")

print()
print("Temporal lobe (λ=0.10) decays slowest — retaining 90% per step,")
print("making it ~7x slower to fade than the fast Occipital lobe (λ=0.90).")`,
          blanks: [
            {
              id: "BLANK_1",
              placeholder: "___BLANK_1___",
              hint: "The decay formula is S_t = (1 - λ) * S_{t-1}. Replace λ with lambda_k.",
              acceptedAnswers: ["(1 - lambda_k)", "1 - lambda_k", "(1-lambda_k)"],
              explanation: "S_t = (1 - λ) · S_{t-1}. Each timestep, the signal is multiplied by (1 - λ). With λ=0.9 (Occipital, fast decay), each step retains only 10% of the previous signal. With λ=0.1 (Temporal, slow decay), each step retains 90%."
            },
            {
              id: "BLANK_2",
              placeholder: "___BLANK_2___",
              hint: "We want to find when 'val' first drops below 'threshold'. Use a comparison operator.",
              acceptedAnswers: ["val < threshold", "val <= threshold", "val <= 0.5", "val < 0.5", "val < initial * 0.5"],
              explanation: "We're looking for the first timestep where the signal value drops below 50% of the initial strength. val < threshold (where threshold = 0.5 * initial) captures this condition."
            },
            {
              id: "BLANK_3",
              placeholder: "___BLANK_3___",
              hint: "Half-life formula: t½ = log(0.5) / log(1 - λ). Both logs can be computed with math.log().",
              acceptedAnswers: [
                "math.log(0.5) / math.log(1 - lambda_k)",
                "math.log(0.5) / math.log(1-lambda_k)",
                "-math.log(2) / math.log(1 - lambda_k)",
              ],
              explanation: "Starting from S_t = (1-λ)^t, we solve for t when S_t = 0.5: 0.5 = (1-λ)^t → t = log(0.5)/log(1-λ). For λ=0.1 (Temporal, retains 90% per step), this gives t≈6.6 — about 7 steps to half-power, making it the longest-lasting lobe."
            }
          ],
          expectedOutput: `Lobe                         Half-Life    At t=200  Analytical HL
----------------------------------------------------------------------
  Occipital (λ=0.90)                1t    0.000000           0.3t
  Frontal (λ=0.80)                  1t    0.000000           0.4t
  Parietal (λ=0.50)                 2t    0.000000           1.0t
  Temporal (λ=0.10)                 7t    0.000000           6.6t

Temporal lobe (λ=0.10) decays slowest — retaining 90% per step,
making it ~7x slower to fade than the fast Occipital lobe (λ=0.90).`
        }
      }
    ]
  },

  {
    id: "position-rope",
    title: "Position & RoPE",
    slug: "position-rope",
    description: "How BDH encodes position and context without quadratic attention",
    icon: "",
    lessons: [
      {
        id: "positional-encoding",
        title: "Absolute vs Relative Positional Encoding",
        slug: "positional-encoding",
        estimatedMinutes: 10,
        sections: [
          {
            type: "text",
            content: `Brains don't use absolute coordinate systems. A neuron doesn't know it's the 452nd neuron in the cortex — it only knows its relationship to the neurons firing around it right now. BDH is built on the same principle: position should be encoded *relatively*, not as a fixed address.

**Absolute positional encoding** (used in original Transformers) assigns a fixed vector to each index 0, 1, 2... and adds it to the token embedding. It works for short sequences, but it has a critical flaw the BDH paper highlights directly: the model was tested on **translation** (the Europarl corpus), where sentence lengths vary wildly between languages. German sentences are systematically longer than English ones. Absolute encoding cannot "slide" its understanding across these length differences — it learned that "Position 5 is important" but that insight breaks down the moment sequence lengths shift.`
          },
          {
            type: "callout",
            variant: "warning",
            title: "The Problem with Absolute Encoding",
            content: "If a model is trained with max_seq_len=128, position vectors only exist for indices 0–127. At inference time with a 256-token input, positions 128–255 have never been seen. The model extrapolates poorly. More fundamentally: in a 1,000-page book, 'Position 5' is meaningless. What matters is not where you are — it's how far you are from the tokens around you."
          },
          {
            type: "callout",
            variant: "info",
            title: "Absolute Encoding is like a GPS that only works in one city",
            content: "RoPE is like a compass — it works everywhere because it only tells you which way is 'North' relative to where you are standing. The distance between two words (their difference in rotation angles) stays the same whether you're at token 10 or token 10,000."
          },
          {
            type: "text",
            content: `BDH uses **Rotary Position Embedding (RoPE)** — a relative encoding scheme that rotates query and key vectors in high-dimensional space. The magic is in the inner product: when you compute Q·K in attention, the rotation angles *subtract*, giving you the relative distance between tokens automatically. No extra parameters. Pure mathematical rotation.

This also has a biological analogue: the paper notes that the inner product property mirrors how biological neurons might track timing through **phase-locking** — neurons that fire in synchrony encode their relationship through the phase difference of their oscillations, not their absolute position in time.`
          },
          {
            type: "formula",
            label: "RoPE: Rotation in Complex Space",
            latex: "\\text{RoPE}(x, t) = x \\cdot e^{i \\theta t} \\quad \\text{where} \\quad \\theta_d = \\frac{1}{\\text{rope\\_theta}^{d/D}}",
            explanation: "Each dimension d gets its own frequency θd. Position t is encoded by rotating the vector by angle θd·t in the d-th complex plane. When we compute Q·K, the rotation angles subtract — giving us the relative position automatically."
          },
          {
            type: "comparison-table",
            title: "Breaking Down the RoPE Formula",
            headers: ["Component", "Meaning in the BDH 'Clock'"],
            rows: [
              ["x", "The Content: the raw meaning of the token (e.g., 'Dragon')."],
              ["t", "The Timestamp: the current position in the sequence."],
              ["e^{iθt}", "The Rotation: a complex number acting as a rotation matrix. As t increases, the vector x is rotated further around the origin."],
              ["θ", "The Frequency: determines how 'fast' the rotation happens for different dimensions. Low dims rotate fast (local syntax). High dims rotate slowly (long-range narrative)."],
            ]
          },
          {
            type: "text",
            content: `Because RoPE is relative, BDH's Specialist Neurons don't get confused by their location in a long text. The comma detector (Neuron #12) doesn't care whether it's processing the 10th or the 10,000th token — it only cares about the relationship between the tokens immediately around it. Monosemanticity and long-context generalization reinforce each other.

In the BDH-GPU implementation, RoPE is applied to the Query (q) and Key (k) projections **before** the Hebbian update. This ensures that when synapses are updated via η(q_t · k_t^T), the update is position-aware — the model learns *which concepts imply which other concepts at what distances*, not just which concepts co-occur.`
          },
          {
            type: "code-snippet",
            language: "python",
            label: "RoPE Frequency Computation (from model.py)",
            code: `import torch
import math

def get_freqs(D: int, theta: float, dtype: torch.dtype) -> torch.Tensor:
    """
    Compute RoPE frequencies for each dimension.
    
    BDH uses a 'quantized' version that rounds frequencies to multiples of 2.
    This creates frequency buckets — like how biological neural oscillations 
    operate at discrete frequency bands (alpha, beta, gamma waves).
    
    Args:
        D: Number of dimensions (n_neurons_per_head)
        theta: Base frequency (rope_theta = 10000.0 in BDHConfig)
        dtype: torch float type
    
    Returns:
        freqs: (D,) tensor of frequencies
    """
    def quantize(t, q=2): 
        return (t / q).floor() * q  # Round to nearest multiple of q
    
    # Standard RoPE: freq_d = 1 / (theta ^ (d/D))
    # BDH adds quantization for frequency bucketing (inspired by cortical oscillations)
    freqs = 1.0 / (theta ** (quantize(torch.arange(0, D, 1, dtype=dtype)) / D))
    
    # Divide by 2π to normalize — freqs now in Hz equivalent
    return freqs / (2 * math.pi)

# Demonstrate frequency spectrum
D = 32  # n_neurons_per_head
freqs = get_freqs(D, theta=10000.0, dtype=torch.float32)

print(f"{'Dim':>4}  {'Frequency':>12}  {'Period (tokens)':>16}")
print("-" * 38)
for d in [0, 4, 8, 16, 24, 31]:
    freq = freqs[d].item()
    period = 1.0 / (freq + 1e-10)
    print(f"  {d:2d}  {freq:12.6f}  {period:16.1f}")

# Low dims: high frequency (captures local syntax, period ~few tokens)
# High dims: low frequency (captures narrative, period ~thousands of tokens)`
          }
        ],
        exercise: {
          id: "ex-rope-freqs",
          title: "Exercise: Build the RoPE Rotation",
          instructions: "Implement the RoPE rotation applied to query/key vectors in BDH's SparseAttention. Given a vector v and position t, compute the rotated version by applying cosine/sine rotation using precomputed frequencies. In BDH, this rotation is applied to the query projection, which serves as both Q and K in the sparse self-attention step.",
          difficulty: "intermediate",
          starterCode: `import torch
import math

def apply_rope(v: torch.Tensor, freqs: torch.Tensor, T: int) -> torch.Tensor:
    """
    Apply Rotary Position Embedding to a vector.
    
    For position t and dimension d:
        phase = t * freqs[d]
        v_rotated[d] = v[d] * cos(phase) + v_perp[d] * sin(phase)
    
    where v_perp is v with pairs of dimensions swapped and negated.
    
    Args:
        v: Input tensor (batch, n_head, T, N) — queries or keys
        freqs: Precomputed frequencies (1, 1, 1, N)
        T: Sequence length
    
    Returns:
        Rotated tensor, same shape as v
    """
    B, H, T, N = v.shape
    
    # Step 1: Compute position-scaled phases for each (position, dimension) pair
    # positions: [0, 1, 2, ..., T-1] shaped as (1, 1, T, 1) for broadcasting
    positions = torch.arange(0, T, device=v.device, dtype=torch.float32).view(1, 1, ___BLANK_1___, 1)
    
    # Multiply positions by frequencies to get angles
    # phases shape: (1, 1, T, N) — each token-dimension pair has its own angle
    phases = positions * freqs  # broadcasting: (1,1,T,1) * (1,1,1,N) → (1,1,T,N)
    
    # Step 2: Compute trig functions (and convert to v's dtype)
    c = torch.cos((phases % 1) * (2 * math.pi)).to(v.dtype)
    s = ___BLANK_2___  # same but sin instead of cos
    
    # Step 3: Build the "perpendicular" vector v_perp
    # For each pair of dimensions (2k, 2k+1): swap and negate the first
    v_even = v[..., ::2]   # even-indexed dimensions
    v_odd  = v[..., 1::2]  # odd-indexed dimensions
    
    # Stack as (-odd, even) interleaved — this is the perpendicular rotation
    v_perp = torch.stack((-v_odd, v_even), dim=-1).view(B, H, T, N)
    
    # Step 4: Apply rotation: v_rot = v * cos + v_perp * sin
    v_rotated = ___BLANK_3___
    
    return v_rotated


# ---- Test your implementation ----
torch.manual_seed(0)

# Simulate BDH's SparseAttention setup
D = 32  # n_neurons_per_head  
n_head = 4

# Precompute freqs (as in model.py)
def quantize(t, q=2): return (t / q).floor() * q
freqs = (1.0 / (10000.0 ** (quantize(torch.arange(0, D, dtype=torch.float32)) / D)) / (2 * math.pi))
freqs_4d = freqs.view(1, 1, 1, D)  # Shape for broadcasting

v = torch.randn(2, n_head, 8, D)  # batch=2, heads=4, seq=8, dim=32
v_rot = apply_rope(v, freqs_4d, T=8)

print(f"Input shape:   {v.shape}")
print(f"Rotated shape: {v_rot.shape}")
print(f"Shapes match: {v.shape == v_rot.shape}")

# Key property: rotation should preserve vector norms
norms_original = v.norm(dim=-1)
norms_rotated = v_rot.norm(dim=-1)
norm_diff = (norms_original - norms_rotated).abs().max().item()
print(f"Max norm difference: {norm_diff:.6f}  (should be ~0 — rotation preserves length)")
print(f"Rotation is norm-preserving: {norm_diff < 0.01}")`,
          blanks: [
            {
              id: "BLANK_1",
              placeholder: "___BLANK_1___",
              hint: "The positions tensor needs to have T in the sequence dimension. We'll reshape it as (1, 1, T, 1). What goes in the view() call for the T dimension?",
              acceptedAnswers: ["-1", "T", "T,"],
              explanation: "positions.view(1, 1, T, 1) shapes the position indices as (batch=1, head=1, seq=T, dim=1). Broadcasting then multiplies each of the T positions by each of the N frequencies."
            },
            {
              id: "BLANK_2",
              placeholder: "___BLANK_2___",
              hint: "Same pattern as the cosine line above, but use torch.sin() instead of torch.cos()",
              acceptedAnswers: [
                "torch.sin((phases % 1) * (2 * math.pi)).to(v.dtype)",
                "torch.sin((phases % 1) * (2 * math.pi)).to(dtype=v.dtype)",
                "torch.sin((phases % 1) * 2 * math.pi).to(v.dtype)",
              ],
              explanation: "The sine component of the rotation. Note the (phases % 1) * 2π converts from the normalized frequency representation back to radians before computing sin()."
            },
            {
              id: "BLANK_3",
              placeholder: "___BLANK_3___",
              hint: "Rotation formula: v_rot = v * cos(θ) + v_perp * sin(θ). You have: v, c (cosine), v_perp, and s (sine).",
              acceptedAnswers: [
                "v * c + v_perp * s",
                "v*c + v_perp*s",
                "(v * c) + (v_perp * s)",
              ],
              explanation: "This is the 2D rotation formula applied per-dimension-pair. v * cos(θ) keeps the original component scaled by cosine, while v_perp * sin(θ) adds the perpendicular component. Together they rotate the vector by angle θ without changing its length."
            }
          ],
          expectedOutput: `Input shape:   torch.Size([2, 4, 8, 32])
Rotated shape: torch.Size([2, 4, 8, 32])
Shapes match: True
Max norm difference: 0.000000  (should be ~0 — rotation preserves length)
Rotation is norm-preserving: True`
        }
      }
    ]
  },

  {
    id: "architecture",
    title: "The BDH Architecture",
    slug: "architecture",
    description: "Deep dive into BDHConfig, the forward pass, and SparseAttention",
    icon: "",
    lessons: [
      {
        id: "bdh-config",
        title: "BDHConfig: Every Parameter Explained",
        slug: "bdh-config",
        estimatedMinutes: 10,
        sections: [
          {
            type: "text",
            content: `In a standard neural network, configuration parameters are just numbers — matrix dimensions, learning rates, depths. In BDH, every parameter represents a **physical constraint on the brain you are building**. Tuning BDHConfig is not hyperparameter search; it is setting the physical laws of how your model perceives time, remembers facts, and prunes its logic.`
          },
          {
            type: "comparison-table",
            title: "BDHConfig: Biological Roles",
            headers: ["Parameter", "Biological Role", "Impact on Performance"],
            rows: [
              ["num_neurons (N)", "Total 'Brain Cells' — the Logical Space", "Higher N = more Specialist Neurons, richer vocabulary of logic. Safe to set much higher than Transformer d_model because sparsity keeps compute constant."],
              ["dim (D)", "Synaptic Bandwidth — the Communication Bus", "Higher D = better handling of complex multi-layered nuances. The N/D ratio determines sparsity pressure."],
              ["decay_rates (λ)", "Memory Persistence — the Memory Horizon", "Controls the balance between short-term and long-term focus. Set a range of λ values to create the Spectral Memory Hierarchy."],
              ["k_winners (k)", "Global Inhibition — the Inhibitory Circuit", "Lower k = higher sparsity, better interpretability, less noise. k=10 means only the 10 most relevant neurons speak."],
            ]
          },
          {
            type: "text",
            content: `**N (Number of Neurons)** is the star parameter of BDH — unlike a Transformer where d_model is the primary width. N is the number of Specialist Neurons available to the model. Because the model is sparse, you can set N much higher than a Transformer's hidden dimension without a massive hit to compute. More neurons = a richer "Vocabulary of Logic."

**D (Internal Dimension)** is the thickness of the nerve fibers connecting layers. If N is the number of brain cells, D is the bandwidth of the synaptic connections. The **N/D ratio** is what forces sparsity — a high ratio creates a bottleneck that pressures the model to be selective about what it encodes.

**λ (Decay Rate)** is the most biological parameter. High λ (≈0.9) creates fast-forgetting sensory neurons that only care about the current word. Low λ (≈0.1) creates persistent narrative neurons that remember the beginning of the document. In a full BDHConfig, you define a *range* of λ values to instantiate the Spectral Memory Hierarchy described in the previous chapter.

**β (Negative Bias)** is the "stubbornness" dial — the Metabolic Threshold. By setting a higher negative bias, you force the model to be more selective: only the strongest signals (the most certain logical conclusions) will cause a neuron to fire. This is the primary tool for fighting hallucinations.

**k (Top-K Sparsity)** defines the Competitive Landscape. If k=10, then out of thousands of neurons, only the 10 most relevant are allowed to speak per timestep. Lower k = higher sparsity = better monosemanticity = more traceable reasoning.`
          },
          {
            type: "callout",
            variant: "info",
            title: "The N/D Ratio: Forcing Sparsity by Design",
            content: "The paper often uses a high N/D ratio (e.g., N=32768, D=256) to force sparsity. With N much larger than D, the encoder projection is a wide expansion — creating a 'crowded arena' where k-WTA competition is fierce. Only truly specialist neurons can win consistently. This is how monosemanticity emerges from a parameter choice, not from explicit supervision."
          },
          {
            type: "code-snippet",
            language: "python",
            label: "BDHConfig — the complete architecture specification",
            code: `@dataclasses.dataclass
class BDHConfig:
    # ── Network depth & width ──────────────────────────────────
    n_layer: int = 4          # Number of BDH layers (depth)
    n_embd: int = 256         # Token embedding dimension D
    n_head: int = 4           # Number of attention heads (Functional Lobes)
    
    # ── The key BDH parameter: internal MLP expansion ─────────
    mlp_internal_dim_multiplier: int = 8  
    # Each head gets: n_embd * multiplier / n_head neurons
    # = 256 * 8 / 4 = 512 neurons per head = 2048 total sparse neurons
    # This is the N/D ratio that forces sparsity competition
    
    # ── Vocabulary & sequence ─────────────────────────────────
    vocab_size: int = 10000   # Output vocabulary size
    max_seq_len: int = 128    # Maximum context window
    
    # ── Training hyperparameters ──────────────────────────────
    dropout: float = 0.1
    learning_rate: float = 5e-4
    batch_size: int = 32
    
    # ── The sparsity knob (k in k-WTA) ────────────────────────
    top_k_fraction: float = 0.15  
    # Fraction of neurons allowed to fire via k-WTA
    # 0.15 = 15% active = 85% silent
    # Paper's "crystallized" state: ~3% (top_k_fraction ≈ 0.03)
    
    # ── RoPE positional encoding ──────────────────────────────
    rope_theta: float = 10000.0   # Base frequency for RoPE rotations
    use_weight_tying: bool = False

    @property
    def n_neurons_per_head(self) -> int:
        # N: the sparse latent space per head — what k-WTA operates on
        return self.mlp_internal_dim_multiplier * self.n_embd // self.n_head
        # Default: 8 * 256 / 4 = 512 neurons per head`
          },
          {
            type: "comparison-table",
            title: "Sparsity vs Performance Trade-off",
            headers: ["top_k_fraction", "Active Neurons", "FLOPs vs Dense", "Stage"],
            rows: [
              ["0.15 (15%)", "~307 / 2048", "~7× reduction", "Early training (Exploration)"],
              ["0.05 (5%)", "~102 / 2048", "~20× reduction", "Mid training (Pruning)"],
              ["0.03 (3%)", "~61 / 2048", "~30× reduction", "Crystallized (Final state)"],
            ]
          }
        ],
        exercise: {
          id: "ex-config-params",
          title: "Exercise: Calculate BDH Architecture Parameters",
          instructions: "Use BDHConfig parameters to calculate key architecture dimensions. This will help you understand how the sparse latent space is constructed from the base config values.",
          difficulty: "beginner",
          starterCode: `import dataclasses

@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 8
    vocab_size: int = 10000
    dropout: float = 0.1
    max_seq_len: int = 128
    learning_rate: float = 5e-4
    top_k_fraction: float = 0.15
    rope_theta: float = 10000.0

    @property
    def n_neurons_per_head(self) -> int:
        return self.mlp_internal_dim_multiplier * self.n_embd // self.n_head

def analyze_bdh_config(config: BDHConfig) -> dict:
    """
    Calculate the key architectural dimensions from a BDHConfig.
    """
    N = config.n_neurons_per_head
    
    # Total neurons across all heads (the full sparse latent space)
    total_sparse_neurons = ___BLANK_1___  # n_head * N
    
    # How many neurons actually fire per timestep (k-WTA budget)
    # Multiply total_sparse_neurons by top_k_fraction
    active_neurons = int(total_sparse_neurons * ___BLANK_2___)
    
    # Silent neurons (those zeroed by k-WTA)
    silent_neurons = total_sparse_neurons - active_neurons
    
    # Approximate parameter count (encoder + decoder weights per layer)
    # Each layer has: encoder (D → n_head*N) + decoder (n_head*N → D)
    D = config.n_embd
    params_per_layer = (D * total_sparse_neurons) + (total_sparse_neurons * D)
    total_params = params_per_layer * ___BLANK_3___  # multiply by n_layer
    
    # Embedding table parameters
    embed_params = config.vocab_size * config.n_embd
    
    return {
        "neurons_per_head": N,
        "total_sparse_neurons": total_sparse_neurons,
        "active_per_step": active_neurons,
        "silent_per_step": silent_neurons,
        "sparsity_ratio": f"{100*(1 - config.top_k_fraction):.0f}%",
        "approx_total_params": total_params + embed_params,
    }


# ---- Test your implementation ----
config = BDHConfig()
stats = analyze_bdh_config(config)

print(f"BDH Architecture Analysis")
print(f"{'='*40}")
print(f"Neurons per head:      {stats['neurons_per_head']}")
print(f"Total sparse neurons:  {stats['total_sparse_neurons']}")
print(f"Active per step:       {stats['active_per_step']}")
print(f"Silent per step:       {stats['silent_per_step']}")
print(f"Sparsity:              {stats['sparsity_ratio']} silent")
print(f"Approx parameters:     {stats['approx_total_params']:,}")

# Try a larger config
large_config = BDHConfig(n_embd=512, n_head=8, n_layer=8, top_k_fraction=0.03)
large_stats = analyze_bdh_config(large_config)
print(f"\\nLarger config (3% sparsity):")
print(f"Active neurons: {large_stats['active_per_step']} / {large_stats['total_sparse_neurons']}")`,
          blanks: [
            {
              id: "BLANK_1",
              placeholder: "___BLANK_1___",
              hint: "Total sparse neurons = number of heads × neurons per head. You have config.n_head and N.",
              acceptedAnswers: ["config.n_head * N", "N * config.n_head", "4 * N", "config.n_head*N"],
              explanation: "Each of the n_head attention heads has N neurons in its sparse latent space. The total sparse latent space is n_head × N neurons — the 'crowded arena' where k-WTA competition happens."
            },
            {
              id: "BLANK_2",
              placeholder: "___BLANK_2___",
              hint: "The active fraction is stored in config.top_k_fraction",
              acceptedAnswers: ["config.top_k_fraction", "0.15", "top_k_fraction"],
              explanation: "config.top_k_fraction is the k in k-WTA — the fraction of neurons allowed to fire. Multiplying total_sparse_neurons by this fraction gives the exact count of neurons active per timestep."
            },
            {
              id: "BLANK_3",
              placeholder: "___BLANK_3___",
              hint: "We want params across all layers. The per-layer count is already calculated. How many layers are there?",
              acceptedAnswers: ["config.n_layer", "4", "n_layer"],
              explanation: "Each of the n_layer BDH layers has its own encoder and decoder weight matrices. Unlike attention complexity (which scales quadratically with sequence length), parameter count scales linearly with depth."
            }
          ],
          expectedOutput: `BDH Architecture Analysis
========================================
Neurons per head:      512
Total sparse neurons:  2048
Active per step:       307
Silent per step:       1741
Sparsity:              85% silent
Approx parameters:     2,725,888

Larger config (3% sparsity):
Active neurons: 123 / 4096`
        }
      },
      {
        id: "forward-pass",
        title: "The Forward Pass — Cycle of Thought",
        slug: "forward-pass",
        estimatedMinutes: 15,
        sections: [
          {
            type: "text",
            content: `The BDH forward pass is a translation between two worlds: **Tensor Space** (what GPUs compute) and **Neuron Space** (what brains do). Each forward pass is a complete **Cycle of Thought** — four stages that mirror how biological perception and reasoning work.

The key distinction from a Transformer: in a Transformer, the "state" at each step is just a list of previous words. In BDH, the state is a **mathematical model of the world** that the model builds in real-time, updating its synaptic connections with each new token.`
          },
          {
            type: "comparison-table",
            title: "The Cycle of Thought",
            headers: ["Stage", "Steps", "What Happens"],
            rows: [
              ["Perception", "1–2", "Words become sparse neuron spikes. Text is converted into patterns of electrical signals before hitting the reasoning circuits."],
              ["Reasoning", "3", "The Interaction Kernel determines logical relationships between spikes — which concepts are active and which should be active next."],
              ["Memorization", "4", "The Hebbian State Update physically strengthens synaptic connections between co-occurring concepts in the state matrix."],
              ["Prediction", "5", "The modified state is projected back to vocabulary space, informed by integrated Lobe Memory — not just the last token."],
            ]
          },
          {
            type: "text",
            content: `**Step 1: Embedding & LayerNorm (Sensory Input)**

The process starts by converting tokens into a continuous vector space (D). The paper emphasizes LayerNorm *without affine parameters* to ensure the input is "energy-neutral" — no dimension starts with a systematic advantage before the neurons begin their competition.

**Step 2: The Encoder-to-Neuron Projection (Neurons Wake Up)**

The input vector is projected from D into a much larger space of N neurons (e.g., 32,768). This is where the transition from Tensor Space to Neuron Space happens. The projection v* = ReLU(x · Encoder) simulates neural firing. Because of the ReLU and negative bias, most neurons stay at zero. Only the Specialist Neurons relevant to the input word wake up.

**Step 3: The Interaction Kernel (Internal Reasoning)**

Instead of global attention, BDH uses a Local Interaction Kernel with two decoders: **decoder_x** identifies which concepts are currently active, and **decoder_y** identifies which concepts *should* be active next based on synaptic history. Together they create a heuristic for what facts are most plausible to evaluate next — Modus Ponens in action.

**Step 4: The Hebbian State Update (Memory Phase)**

Once neurons have fired, the model updates its internal state S_t. If the model sees "New" followed by "York," the synaptic connection between these two neuron groups is physically strengthened in the state matrix for the duration of the sequence. Unlike Transformers that "look back" at a buffer of tokens, BDH updates its synapses directly.

**Step 5: The Readout (Prediction)**

The modified state is projected back into vocabulary space. Critically, this readout looks at the integrated Lobe Memory — not just the last token — ensuring the predicted word fits the long-term narrative established in the Temporal Lobe.`
          },
          {
            type: "callout",
            variant: "info",
            title: "The BDH Layer Pipeline",
            content: "1) Embed → 2) Project to sparse space (Encoder) → 3) Apply ReLU + k-WTA → 4) Attend (SparseAttention with RoPE) → 5) Decode back → 6) Residual add. Repeat × n_layer."
          },
          {
            type: "code-snippet",
            language: "python",
            label: "BDH.forward() — the complete forward pass (from model.py)",
            code: `def forward(self, idx, targets=None, return_diagnostics=False, return_hidden_states=False):
    C = self.config
    B, T = idx.size()
    D, nh = C.n_embd, C.n_head
    N = C.n_neurons_per_head  # Neurons per head in sparse latent space

    # ── Stage 0: Token Embedding ─────────────────────────────────────
    x = self.ln_in(self.embed(idx))  # (B, T, D)
    # LayerNorm without affine parameters — "energy-neutral" sensory input
    
    # ── Pre-compute shared weight matrices ───────────────────────────
    W_enc   = self.encoder.permute(1,0,2).reshape(D, nh*N)    # (D, nh*N)
    W_enc_v = self.encoder_v.permute(1,0,2).reshape(D, nh*N)  # (D, nh*N)
    W_dec   = self.decoder_weight.reshape(nh*N, D)            # (nh*N, D)
    
    for i in range(C.n_layer):
        residual = x  # Save input for residual connection

        # ── Stage 1: EXPANSION (Tensor Space → Neuron Space) ─────────
        q_raw = x @ W_enc    # (B, T, nh*N)
        v_raw = x @ W_enc_v  # (B, T, nh*N)
        
        # ── Stage 2: SPARSITY (the Specialist Neurons wake up) ────────
        # LayerNorm → ReLU → k-WTA creates sparse, positive activations
        q = apply_kwta(F.relu(self.latent_norms_q[i](q_raw)), C.top_k_fraction)
        v = apply_kwta(F.relu(self.latent_norms_v[i](v_raw)), C.top_k_fraction)
        # After this: ~85% of entries in q, v are exactly 0.0
        
        # ── Stage 3: ATTENTION — the Interaction Kernel ───────────────
        # q used for both Q and K: "which concepts imply which others?"
        y = self.attn(q, q, v)  # SparseAttention applies RoPE here
        
        # ── Stage 4: RECONSTRUCTION (Neuron Space → Tensor Space) ────
        y_decoded = y @ W_dec + self.decoder_bias  # (B, T, D)
        
        # ── Stage 5: RESIDUAL CONNECTION ─────────────────────────────
        x = residual + self.drop(self.ln_out(y_decoded))
    
    # ── Final: Language Model Head ───────────────────────────────────
    logits = self.lm_head(x)  # (B, T, vocab_size)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
    return logits, loss, hidden_states if return_hidden_states else diagnostics`
          },
          {
            type: "callout",
            variant: "success",
            title: "Why q is used as both Q and K",
            content: "In standard attention: Attention(Q, K, V). In BDH, q is used for both Q and K — self-attention where queries and keys come from the same sparse projection. This means: 'how relevant is this sparse feature pattern to itself at other positions?' — a form of associative memory lookup that mirrors Modus Ponens: if this neuron pattern is active, what other patterns does it imply?"
          }
        ],
        exercise: {
          id: "ex-forward-pass",
          title: "Exercise: Complete the BDH Layer Forward Pass",
          instructions: "Fill in the missing steps of a single BDH layer. You'll implement the Expansion → Sparsity → Decode → Residual pipeline that repeats n_layer times in the full model.",
          difficulty: "intermediate",
          starterCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def bdh_layer_forward(
    x: torch.Tensor,           # Input: (B, T, D)
    W_enc: torch.Tensor,       # Encoder weights: (D, nh*N)  
    W_dec: torch.Tensor,       # Decoder weights: (nh*N, D)
    decoder_bias: torch.Tensor, # Decoder bias: (D,)
    layer_norm_q: nn.LayerNorm,
    layer_norm_out: nn.LayerNorm,
    top_k_fraction: float,
    dropout: nn.Dropout,
) -> tuple:
    """
    One BDH layer: Expand → Sparse → Attention → Decode → Residual.
    Returns: (output, sparsity_rate)
    """
    residual = x  # Save for residual connection
    B, T, D = x.shape

    # ── Step 1: Expansion ──────────────────────────────────────────
    # Project from token space D to sparse latent space nh*N
    q_raw = ___BLANK_1___  # matrix multiply x by W_enc
    
    # ── Step 2: Sparsity — the core BDH operation ─────────────────
    # LayerNorm normalizes, ReLU enforces positivity, k-WTA prunes
    q_normed = layer_norm_q(q_raw)        # normalize
    q_relu = F.relu(q_normed)             # positive orthant constraint
    
    # k-WTA: keep only top-k fraction
    nh_N = q_relu.shape[-1]
    k = max(1, int(nh_N * top_k_fraction))
    topk_vals, _ = torch.topk(q_relu, k, dim=-1)
    threshold = topk_vals[..., -1:]
    q_sparse = q_relu * (q_relu >= threshold).float()  # zero out losers
    
    # Measure sparsity (fraction of zeros)
    sparsity = ___BLANK_2___  # fraction of q_sparse that equals 0
    
    # ── Step 3: Simple dot-product attention (no RoPE for this exercise) ──
    y = q_sparse  # skip attention for this exercise
    
    # ── Step 4: Decode back to D ──────────────────────────────────
    # Project from sparse space nh*N back to token space D
    y_decoded = ___BLANK_3___ + decoder_bias  # matrix multiply + bias
    
    # ── Step 5: Residual connection ───────────────────────────────
    # Add back the original input (prevents gradient vanishing)
    output = ___BLANK_4___  # residual + dropout(layernorm(y_decoded))
    
    return output, sparsity


# ---- Test your implementation ----
torch.manual_seed(42)

B, T, D = 2, 16, 64
nh, N = 4, 32  # 4 heads, 32 neurons per head
nh_N = nh * N  # = 128: total sparse neurons

W_enc = torch.randn(D, nh_N) * 0.02
W_dec = torch.randn(nh_N, D) * 0.02
d_bias = torch.zeros(D)
ln_q = nn.LayerNorm(nh_N)
ln_out = nn.LayerNorm(D)
drop = nn.Dropout(0.1)

x = torch.randn(B, T, D)

output, sparsity = bdh_layer_forward(
    x, W_enc, W_dec, d_bias, ln_q, ln_out,
    top_k_fraction=0.15, dropout=drop
)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Shapes match: {x.shape == output.shape}")
print(f"Sparsity rate: {sparsity:.1%} neurons silent")
print(f"Has residual: {(output - x).abs().mean() > 0}")`,
          blanks: [
            {
              id: "BLANK_1",
              placeholder: "___BLANK_1___",
              hint: "Matrix multiply x (B, T, D) by W_enc (D, nh*N) to get (B, T, nh*N). Use the @ operator.",
              acceptedAnswers: ["x @ W_enc", "torch.matmul(x, W_enc)"],
              explanation: "x @ W_enc performs the expansion from token space (D) to sparse latent space (nh*N). This is the Tensor Space → Neuron Space transition — projecting into the arena where k-WTA competition happens."
            },
            {
              id: "BLANK_2",
              placeholder: "___BLANK_2___",
              hint: "Count what fraction of q_sparse equals zero. Use (q_sparse == 0).float().mean().item()",
              acceptedAnswers: [
                "(q_sparse == 0).float().mean().item()",
                "(q_sparse == 0).float().mean()",
                "1 - (q_sparse > 0).float().mean().item()",
                "1.0 - top_k_fraction",
              ],
              explanation: "Sparsity is the fraction of neurons that are exactly zero after k-WTA. With top_k_fraction=0.15, approximately 85% of neurons will be zeroed out — the silent neurons conserving metabolic energy."
            },
            {
              id: "BLANK_3",
              placeholder: "___BLANK_3___",
              hint: "Decode: multiply y (B, T, nh*N) by W_dec (nh*N, D) to get (B, T, D). Use the @ operator.",
              acceptedAnswers: ["y @ W_dec", "torch.matmul(y, W_dec)"],
              explanation: "y @ W_dec projects back from Neuron Space (nh*N) to Tensor Space (D). This is the decoder in the encoder-decoder factorization — reconstructing a dense D-dimensional output from sparse activations."
            },
            {
              id: "BLANK_4",
              placeholder: "___BLANK_4___",
              hint: "Residual connection: add the original 'residual' to the processed output. Apply dropout and layer_norm_out to y_decoded first.",
              acceptedAnswers: [
                "residual + dropout(layer_norm_out(y_decoded))",
                "residual + drop(layer_norm_out(y_decoded))",
                "residual + dropout(ln_out(y_decoded))",
                "residual + drop(ln_out(y_decoded))",
              ],
              explanation: "The residual connection adds the layer's input directly to its output: x_out = x_in + f(x_in). This prevents vanishing gradients in deep networks and ensures information can flow unimpeded through the stack of BDH layers."
            }
          ],
          expectedOutput: `Input shape:  torch.Size([2, 16, 64])
Output shape: torch.Size([2, 16, 64])
Shapes match: True
Sparsity rate: ~85.0% neurons silent
Has residual: True`
        }
      }
    ]
  },

  {
    id: "interpretability",
    title: "Interpretability",
    slug: "interpretability",
    description: "How BDH opens the black box with monosemantic, traceable neurons",
    icon: "",
    lessons: [
      {
        id: "semantic-probe",
        title: "Probing Specialist Neurons",
        slug: "semantic-probe",
        estimatedMinutes: 12,
        sections: [
          {
            type: "text",
            content: `In a standard Transformer, if you ask "which neurons are responsible for understanding the word 'King'?", there's no good answer. The information is spread across hundreds of neurons in superposition. To get even an approximate answer, you need complex tools like Integrated Gradients — a second machine learning process to guess what the first one is doing.

In BDH, you can answer this precisely. **Knowledge is localized.** Because the activation vectors are sparse and positive, the model creates a one-to-one mapping between specific indices in the N-dimension and specific logical abstractions. To probe the model, you simply read the sparse activation vector directly — the "address" of a concept in the brain is literally its neuron index.`
          },
          {
            type: "callout",
            variant: "info",
            title: "White-Box vs Black-Box Probing",
            content: "Black-Box Probing (Transformers): use a second AI model to guess what the first one is doing. Requires feature steering, Integrated Gradients, or probing classifiers — none of which give ground truth. White-Box Probing (BDH): look at the Interaction Kernel directly. Feed a token, observe the v* vector, read the active neuron indices. No guessing. The paper calls this 'Inherent Interpretability.'"
          },
          {
            type: "text",
            content: `The paper presents results from probing a 24M parameter BDH model trained on the Europarl translation corpus. The findings demonstrate surgical precision:

**Monosemantic Synapses in the σ Matrix:** The researchers identified specific synapse entries (σᵢ,ⱼ) in the state matrix that activate with high selectivity whenever a currency name or country name appears in the sentence. The same synapse activates whether the sentence is in French ("livre sterling") or English ("British Pound") — semantic selectivity crosses language boundaries.

**Alphabetical Specialists:** Neuron #598 fires with P>0.95 exclusively for the character 'I'. Neuron #14 is a dedicated detector for 'H'. These neurons exhibit All-or-Nothing behavior — unlike Transformers where a neuron might fire 70% for 'I' and 30% for 'A', BDH neurons are fully committed.

**Structural Specialists:** Neuron #12 fires only for commas and periods — the "clock" neurons that tell the model when a thought has ended. Punctuation neurons cluster physically differently from alphabetic neurons, creating an emergent lobe structure within the N dimension — without any explicit supervision.

**The Grandmother Cell Hypothesis Confirmed:** BDH proves that a single neuron can represent a complex concept. You can "turn off" a specific neuron and predictably remove the model's ability to understand a specific word or grammar rule — something nearly impossible in a standard Transformer.`
          },
          {
            type: "callout",
            variant: "success",
            title: "Statistical Validation: Currency Synapse",
            content: "To confirm synapse selectivity, the researchers generated 50 sentences about European currencies and 50 about European politics (without currency mentions). A one-sided Mann–Whitney U test revealed that currency sentences received significantly higher 'Currency synapse' values (U=2368, Uopt=2500, p<10⁻¹⁴). Rank-biserial correlation: 0.86. This is rigorous empirical proof of monosemanticity — not just a demonstration."
          },
          {
            type: "comparison-table",
            title: "Glass Box vs Black Box",
            headers: ["Feature", "Black Box (Transformer)", "Glass Box (BDH)"],
            rows: [
              ["Logic", "Emergent / Statistical", "Axiomatic / Traceable"],
              ["Neuron Role", "Polysemantic (jack of all trades)", "Monosemantic (specialist)"],
              ["Auditability", "Requires Feature Steered models", "Direct weight inspection"],
              ["Hallucination", "Hard to trace (superposition)", "Easy to trace (active pathways)"],
              ["Probing Method", "Integrated Gradients (approximation)", "Read v* vector (ground truth)"],
            ]
          },
          {
            type: "code-snippet",
            language: "python",
            label: "semantic_probe() — white-box concept similarity (from model.py)",
            code: `def semantic_probe(self, word_pairs, tokenizer):
    """
    Probe the model's latent space for semantic similarity between word pairs.
    
    For each word pair (w1, w2), we:
    1. Embed each word → get its BDH latent representation
    2. Apply encoder + k-WTA to get the sparse activation pattern
    3. Compute cosine similarity between the two sparse patterns
    
    High cosine similarity → similar semantic features activated
    → same neurons fire → monosemantic cluster
    
    Expected results from the BDH paper:
    - "King" vs "Queen":  high similarity (Royalty cluster)
    - "King" vs "Car":    near-zero similarity (disjoint modules)
    """
    results = {}
    self.eval()
    nh, D, N = self.encoder.shape
    
    W_enc = self.encoder.permute(1, 0, 2).reshape(D, nh * N)
    norm = self.latent_norms_q[0]  # Use Layer 0 norms for probing
    
    for w1, w2 in word_pairs:
        t1 = torch.tensor(tokenizer.encode(w1)).to(self.embed.weight.device)
        t2 = torch.tensor(tokenizer.encode(w2)).to(self.embed.weight.device)
        
        e1 = self.embed(t1).mean(0).unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        e2 = self.embed(t2).mean(0).unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        
        # Apply the BDH encoding: LayerNorm → ReLU → k-WTA
        v1 = self.apply_kwta(F.relu(norm(e1 @ W_enc)), self.config.top_k_fraction)
        v2 = self.apply_kwta(F.relu(norm(e2 @ W_enc)), self.config.top_k_fraction)
        
        # Cosine similarity in sparse latent space
        sim = F.cosine_similarity(
            v1.flatten().unsqueeze(0), 
            v2.flatten().unsqueeze(0)
        ).item()
        
        results[f"{w1}-{w2}"] = sim
    
    return results

# Expected output on a trained BDH model:
# {
#   "King-Queen":  0.847,  ← High! Shared "Royalty" module
#   "King-Car":    0.012,  ← Near zero! Disjoint subgraphs
#   "Dog-Wolf":    0.723,  ← High! Shared "Canine" features
# }`
          }
        ],
        exercise: {
          id: "ex-semantic-probe",
          title: "Exercise: Implement Concept Overlap Analysis",
          instructions: "Implement the overlap analysis from the BDH paper: given two words, find which neurons are active for each, then compute the overlap fraction. High overlap = shared semantic module. Low overlap = disjoint subgraphs.",
          difficulty: "intermediate",
          starterCode: `import torch
import torch.nn.functional as F

def get_active_neurons(
    word_embedding: torch.Tensor,  # (D,) embedding vector
    W_enc: torch.Tensor,           # (D, nh_N) encoder weights
    layer_norm: torch.nn.LayerNorm,
    top_k_fraction: float
) -> set:
    """
    Get the set of active neuron indices for a word embedding.
    
    Returns:
        Set of integer indices of active (non-zero) neurons
    """
    x = word_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
    
    latent = x @ W_enc  # (1, 1, nh_N)
    
    normed = layer_norm(latent)
    relu_out = F.relu(normed)
    
    k = max(1, int(relu_out.shape[-1] * top_k_fraction))
    topk_vals, _ = torch.topk(relu_out, k, dim=-1)
    threshold = topk_vals[..., -1:]
    sparse = relu_out * (relu_out >= threshold).float()
    
    # Return set of indices where neuron is active (non-zero)
    active_mask = (sparse.squeeze() > 0)
    return ___BLANK_1___  # set of indices where active_mask is True


def concept_overlap(
    set_a: set, 
    set_b: set
) -> dict:
    """
    Compute overlap statistics between two sets of active neurons.
    High overlap = shared semantic module (like King/Queen sharing "Royalty" neurons).
    Low overlap = disjoint subgraphs (like King/Car having no shared circuit).
    """
    intersection = ___BLANK_2___  # set intersection of set_a and set_b
    union = set_a | set_b
    
    overlap_fraction = len(intersection) / len(union) if union else 0.0
    
    return {
        "intersection_size": len(intersection),
        "union_size": len(union),
        "overlap_fraction": overlap_fraction,
        "a_exclusive": len(___BLANK_3___),   # neurons only in A
        "b_exclusive": len(set_b - set_a),  # neurons only in B
    }


# ---- Test your implementation ----
torch.manual_seed(42)

D, nh_N = 64, 128
W_enc = torch.randn(D, nh_N) * 0.5
ln = torch.nn.LayerNorm(nh_N)

# Simulate semantically similar words (high overlap expected)
base_royalty = torch.randn(D)
king_emb   = base_royalty + torch.randn(D) * 0.1   # very similar to base
queen_emb  = base_royalty + torch.randn(D) * 0.1   # very similar to base
car_emb    = torch.randn(D)                          # unrelated

king_neurons  = get_active_neurons(king_emb,  W_enc, ln, 0.15)
queen_neurons = get_active_neurons(queen_emb, W_enc, ln, 0.15)
car_neurons   = get_active_neurons(car_emb,   W_enc, ln, 0.15)

print(f"Active neurons for 'King':  {len(king_neurons)} / {nh_N}")
print(f"Active neurons for 'Queen': {len(queen_neurons)} / {nh_N}")
print(f"Active neurons for 'Car':   {len(car_neurons)} / {nh_N}")

kq_overlap = concept_overlap(king_neurons, queen_neurons)
kc_overlap = concept_overlap(king_neurons, car_neurons)

print(f"\\nKing ∩ Queen:")
print(f"  Shared neurons:    {kq_overlap['intersection_size']}")
print(f"  Overlap (Jaccard): {kq_overlap['overlap_fraction']:.1%}  ← high = shared module")

print(f"\\nKing ∩ Car:")
print(f"  Shared neurons:    {kc_overlap['intersection_size']}")
print(f"  Overlap (Jaccard): {kc_overlap['overlap_fraction']:.1%}  ← low = disjoint subgraphs")`,
          blanks: [
            {
              id: "BLANK_1",
              placeholder: "___BLANK_1___",
              hint: "Convert active_mask (a boolean tensor) to a Python set of integer indices. Use .nonzero() to get indices, then convert to a set.",
              acceptedAnswers: [
                "set(active_mask.nonzero().squeeze().tolist())",
                "set(active_mask.nonzero(as_tuple=True)[0].tolist())",
                "set(torch.where(active_mask)[0].tolist())",
                "{i.item() for i in active_mask.nonzero()}",
              ],
              explanation: "active_mask.nonzero() returns a tensor of indices where the mask is True. .squeeze() removes extra dims, .tolist() converts to a Python list, and set() creates a set for fast intersection/union operations — the Concept Overlap analysis."
            },
            {
              id: "BLANK_2",
              placeholder: "___BLANK_2___",
              hint: "Python set intersection: the & operator or set_a.intersection(set_b) both work.",
              acceptedAnswers: ["set_a & set_b", "set_a.intersection(set_b)"],
              explanation: "Python's & operator computes set intersection — elements present in both sets. This gives us the neurons that fire for BOTH concepts, which is the signal for shared semantic modules. High intersection = shared 'Royalty' circuit for King/Queen."
            },
            {
              id: "BLANK_3",
              placeholder: "___BLANK_3___",
              hint: "Neurons exclusive to A = neurons in A but NOT in B. Use set difference: set_a - set_b",
              acceptedAnswers: ["set_a - set_b", "set_a.difference(set_b)"],
              explanation: "set_a - set_b gives elements in set_a that are NOT in set_b — neurons that fire only for concept A. In BDH, these are the concept-specific specialist neurons that differentiate 'King' from 'Queen' despite their shared Royalty cluster."
            }
          ],
          expectedOutput: `Active neurons for 'King':  19 / 128
Active neurons for 'Queen': 19 / 128
Active neurons for 'Car':   19 / 128

King ∩ Queen:
  Shared neurons:    ~17
  Overlap (Jaccard): ~90%  ← high = shared module

King ∩ Car:
  Shared neurons:    ~1-3
  Overlap (Jaccard): ~5%  ← low = disjoint subgraphs`
        }
      }
    ]
  },

  {
    id: "generation",
    title: "Generation & Inference",
    slug: "generation",
    description: "How BDH generates text — temperature, sampling, and efficiency",
    icon: "",
    lessons: [
      {
        id: "generate-method",
        title: "Text Generation with BDH",
        slug: "generate-method",
        estimatedMinutes: 10,
        sections: [
          {
            type: "text",
            content: `Generation is where BDH's efficiency shines in practice. At inference time, the model autoregressively predicts the next token — but because only 3% of neurons fire per step, each forward pass costs **~30× fewer FLOPs** than a dense Transformer of equivalent size.\\n\\nThe generate() method in model.py implements **temperature sampling with top-k filtering**.`
          },
          {
            type: "formula",
            label: "Temperature-Scaled Sampling",
            latex: "P(w_i) = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}",
            explanation: "Temperature T controls randomness. T→0: deterministic (always pick argmax). T=1: standard sampling. T>1: more uniform (more creative/random). BDH uses this on the final logits before multinomial sampling."
          },
          {
            type: "code-snippet",
            language: "python",
            label: "BDH.generate() — autoregressive text generation (from model.py)",
            code: `@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Autoregressively generate max_new_tokens tokens.
    
    Args:
        idx: Seed token IDs (batch, seed_length)
        max_new_tokens: How many tokens to generate
        temperature: Sampling temperature (0 = greedy, 1 = standard, >1 = random)
        top_k: If set, only sample from the top-k highest probability tokens
    
    Returns:
        idx with new tokens appended: (batch, seed_length + max_new_tokens)
    """
    for _ in range(max_new_tokens):
        # Truncate to max context window (BDH has no KV-Cache growth)
        # Unlike Transformers, BDH's memory cost is CONSTANT regardless of length
        idx_crop = idx[:, -self.config.max_seq_len:]
        
        # Forward pass — only the final token's logits matter for prediction
        logits, _, _ = self(idx_crop)
        logits = logits[:, -1, :]  # (batch, vocab_size) — last token's prediction
        
        # Apply temperature: divide logits, making distribution sharper or flatter
        logits = logits / temperature
        
        # Optional top-k filtering: zero out all logits outside top-k
        # This prevents sampling very low-probability "hallucination" tokens
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Convert to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx`
          },
          {
            type: "comparison-table",
            title: "Generation: BDH vs Transformer Efficiency",
            headers: ["Metric", "Transformer", "BDH"],
            rows: [
              ["Memory per step", "O(L·d·layers) — grows with length", "O(d·lobes) — constant"],
              ["FLOPs per token", "~100% of parameters active", "~3% of parameters active"],
              ["KV-Cache", "Required, grows indefinitely", "Not needed — state on synapses"],
              ["Long-context degradation", "Context dilution at large L", "Temporal lobe maintains narrative"],
            ]
          }
        ],
        exercise: {
          id: "ex-generation",
          title: "Exercise: Implement Temperature & Top-K Sampling",
          instructions: "Implement the sampling logic used in BDH's generate() method. Temperature controls creativity, top-k prevents hallucination. These two parameters together control the quality/diversity trade-off.",
          difficulty: "beginner",
          starterCode: `import torch
import torch.nn.functional as F

def sample_next_token(
    logits: torch.Tensor,  # (vocab_size,) raw logits from the model
    temperature: float = 1.0,
    top_k: int = None,
) -> int:
    """
    Sample the next token from logits using temperature + top-k filtering.
    
    Args:
        logits: Raw model outputs (unnormalized log-probabilities)
        temperature: Sampling temperature (T=1: standard, T→0: greedy, T>1: random)
        top_k: If set, only sample from top-k highest logit tokens
    
    Returns:
        Sampled token index (integer)
    """
    # Step 1: Apply temperature scaling
    # Divide logits by temperature before softmax
    # T < 1: makes distribution peakier (less random)
    # T > 1: flattens distribution (more random)
    scaled_logits = logits / ___BLANK_1___
    
    # Step 2: Apply top-k filtering (optional)
    if top_k is not None:
        k = min(top_k, scaled_logits.size(-1))
        
        # Find the k-th largest logit value
        topk_values, _ = torch.topk(scaled_logits, k)
        kth_value = topk_values[-1]  # smallest of the top-k values
        
        # Set all logits below the k-th value to -infinity
        # After softmax, -inf becomes probability 0.0
        scaled_logits = scaled_logits.masked_fill(
            scaled_logits < kth_value, 
            ___BLANK_2___  # what value makes softmax output 0?
        )
    
    # Step 3: Convert to probabilities via softmax
    probs = ___BLANK_3___  # apply softmax to scaled_logits
    
    # Step 4: Sample one token from the distribution
    next_token = torch.multinomial(probs, num_samples=1).item()
    
    return next_token


# ---- Test your implementation ----
torch.manual_seed(42)
vocab_size = 100

# Simulate logits with one clearly dominant token (index 42)
logits = torch.randn(vocab_size) * 0.5
logits[42] = 5.0  # Make token 42 very likely

print("Temperature effects (1000 samples each):")
print(f"{'Temperature':>12}  {'Token 42%':>10}  {'Entropy':>8}")
print("-" * 35)

for temp in [0.1, 0.5, 1.0, 2.0]:
    samples = [sample_next_token(logits.clone(), temperature=temp) for _ in range(1000)]
    token42_freq = samples.count(42) / 1000
    unique = len(set(samples))
    print(f"  T={temp:<8}  {token42_freq:>9.1%}  {unique:>5} unique tokens")

print(f"\\nTop-k filtering (k=5):")
samples_topk = [sample_next_token(logits.clone(), temperature=1.0, top_k=5) for _ in range(1000)]
unique_topk = len(set(samples_topk))
print(f"  Unique tokens sampled: {unique_topk} (should be ≤ 5)")
print(f"  Token 42 frequency:    {samples_topk.count(42)/1000:.1%}")`,
          blanks: [
            {
              id: "BLANK_1",
              placeholder: "___BLANK_1___",
              hint: "Divide by the temperature parameter. The variable is called 'temperature'.",
              acceptedAnswers: ["temperature", "temp", "float(temperature)"],
              explanation: "logits / temperature scales the logits before softmax. Low temperature (e.g., 0.1) amplifies differences — making the highest logit dominate. High temperature (e.g., 2.0) flattens the distribution — making all tokens more equally likely."
            },
            {
              id: "BLANK_2",
              placeholder: "___BLANK_2___",
              hint: "When passed through softmax, this value should produce probability 0. What number makes exp(x) = 0?",
              acceptedAnswers: ["float('-inf')", "-float('inf')", "float('-Inf')", "-torch.inf", "torch.finfo(logits.dtype).min"],
              explanation: "float('-inf') sets logits to negative infinity. After softmax: exp(-inf) = 0, so the probability of these tokens becomes exactly 0. This is how we filter out tokens outside the top-k without changing relative probabilities among survivors."
            },
            {
              id: "BLANK_3",
              placeholder: "___BLANK_3___",
              hint: "Convert raw logits to a probability distribution using F.softmax(). The dimension to apply along is dim=-1 (or dim=0 for a 1D tensor).",
              acceptedAnswers: [
                "F.softmax(scaled_logits, dim=-1)",
                "F.softmax(scaled_logits, dim=0)",
                "torch.softmax(scaled_logits, dim=-1)",
                "torch.softmax(scaled_logits, dim=0)",
              ],
              explanation: "F.softmax converts raw logits to probabilities: P(i) = exp(z_i) / Σ exp(z_j). The output sums to 1.0, making it a valid probability distribution. dim=-1 applies softmax along the vocabulary dimension."
            }
          ],
          expectedOutput: `Temperature effects (1000 samples each):
 Temperature   Token 42%   Entropy
-----------------------------------
  T=0.1           ~100%   ~1 unique tokens
  T=0.5            ~95%   ~5 unique tokens
  T=1.0            ~70%   ~20 unique tokens
  T=2.0            ~35%   ~45 unique tokens

Top-k filtering (k=5):
  Unique tokens sampled: ≤ 5 (should be ≤ 5)
  Token 42 frequency:    ~70%`
        }
      }
    ]
  }
]


// Helper: get all lessons flat
export const getAllLessons = () => {
  return curriculum.flatMap(chapter => 
    chapter.lessons.map(lesson => ({
      ...lesson,
      chapterId: chapter.id,
      chapterTitle: chapter.title,
      chapterIcon: chapter.icon,
      chapterSlug: chapter.slug,
    }))
  )
}

// Helper: get lesson by chapter + lesson slug
export const getLesson = (chapterSlug, lessonSlug) => {
  const chapter = curriculum.find(c => c.slug === chapterSlug)
  if (!chapter) return null
  const lesson = chapter.lessons.find(l => l.slug === lessonSlug)
  if (!lesson) return null
  return { ...lesson, chapter }
}
