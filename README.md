# 🐉 The Dragon Hatchling — BDH Neural Interpretability Suite

> An interactive, full-stack visualization platform for exploring the internals of a BDH — Brain-like Dragon Hatchling. Built to make neural network interpretability intuitive, visual, and live.

---

## 🌐 Hosted Demo & Video

| | Link |
|---|---|
| 🔗 **Live Demo** | [https://matriarchic-ungladly-myrta.ngrok-free.dev/] |
| 🎥 **Demo Video** | [your-video-link-here] |

---

## 🖼️ Screenshots

<!-- Replace the placeholders below with your actual screenshots -->

| 3D Brain Visualization | Monosemanticity |
|---|---|
| ![3D Brain](docs/brain.png) | ![Monosemanticity](docs/monosemanticity.png) |

| VS Battle | Studio |
|---|---|
| ![VS Battle](docs/battle.png) | ![Studio](docs/studio.png) |

---

## 🧠 What We Built

The Dragon Hatchling (BDH) is a custom **k-Winner-Take-All sparse transformer** trained from scratch, paired with a full-stack interpretability dashboard. The model enforces biological-style sparsity at every layer — only the top-k neurons fire per forward pass — making it uniquely interpretable. The visualization suite lets you feed any text input and watch in real-time how neurons activate, how concepts are stored, how Hebbian synapses strengthen, and how ROPE positional encoding works. It also includes a live training Studio where you can train specialist models, merge them, and run inference — all from the browser.

---

## 💡 What Insight It Reveals About BDH

- **Monosemanticity:** Individual neurons in BDH reliably activate for specific semantic concepts (e.g. "military", "nature") rather than polysemantic superpositions — a direct consequence of k-WTA sparsity.
- **Sparse Activation Atlas:** At any forward pass, only ~15% of neurons fire, forming clean, interpretable activation clusters.
- **Hebbian Traces:** Synapse strengthening between co-activating neurons can be watched frame-by-frame, showing how the model learns correlations.
- **ROPE Encoding:** Rotary positional embeddings are visualised geometrically — showing how token positions are encoded as rotations in embedding space.
- **ReLU Low-Rank:** Demonstrates how sparse activations implicitly create low-rank structure in weight matrices.
- **VS Battle:** Side-by-side comparison of BDH vs a standard dense transformer on sparsity, concept specialisation, and sample predictions.

---

## 🗺️ Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | **3D Brain** | Live 3D force-graph of neuron activations for any text input |
| `/monosemanticity` | **Monosemanticity** | Radial/scatter/stream views of sparse neuron pulses |
| `/activation-atlas` | **Activation Atlas** | Concept map of what neurons specialise in |
| `/hebbian` | **Hebbian Synapses** | Frame-by-frame synapse strengthening animation |
| `/concepts` | **Concept Memory** | Persistent concept storage across inputs |
| `/rope` | **ROPE** | Rotary positional encoding visualisation |
| `/relu-lowrank` | **ReLU Low-Rank** | Low-rank structure from sparse activations |
| `/learn` | **Learn** | Interactive tutorial on BDH architecture |
| `/comparison` | **VS Battle** | BDH vs standard Transformer head-to-head |
| `/studio` | **Studio** | Live training, merging, fine-tuning, and inference |

---

## 🏗️ Project Structure

```
The_Dragon_Hatchling/
├── src/                        # SvelteKit frontend
│   ├── routes/                 # One folder per page
│   │   ├── +page.svelte        # 3D Brain (home)
│   │   ├── monosemanticity/
│   │   ├── activation-atlas/
│   │   ├── hebbian/
│   │   ├── concepts/
│   │   ├── rope/
│   │   ├── relu-lowrank/
│   │   ├── comparison/
│   │   └── studio/
│   └── lib/                    # Shared components, stores, utils
├── bdh-tutorial/               # Learn page (separate React/Vite app)
├── static/                     # Static assets & pre-generated JSON data
│   └── data/                   # activations.json, battle_data.json, etc.
├── backend/
│   ├── main.py                 # FastAPI server (serves API + built frontend)
│   ├── studio.py               # Studio endpoints (train, merge, finetune, infer)
│   ├── model_new.py            # BDH k-WTA sparse transformer definition
│   ├── custom_tokenizer.py     # BDH tokenizer wrapper
│   ├── tokenizer.json          # Trained tokenizer vocabulary
│   ├── checkpoint_final.pt     # Trained BDH model weights
│   ├── export_all.py           # One-shot data pre-generation script
│   ├── export_activations.py   # Generates activations.json
│   ├── export_battle.py        # Generates battle_data.json
│   ├── export_comparison.py    # Generates comparison_data.json
│   ├── export_explainer.py     # Generates explainer.json
│   └── requirements.txt        # Python dependencies
├── package.json                # Node dependencies & scripts
├── vite.config.ts              # Vite + SvelteKit config
├── svelte.config.js
├── tailwind.config.js
└── tsconfig.json
```

---

## 🚀 How to Run Locally

### Prerequisites

- **Node.js** v18+ and **npm**
- **Python** 3.9+
- **pip**
- (Optional) CUDA-compatible GPU for faster inference

---

### Step 1 — Install frontend dependencies

```bash
# From repo root
npm install
npm --prefix bdh-tutorial install
```

---

### Step 2 — Install backend dependencies

```bash
# Terminal 1
cd backend
pip install -r requirements.txt
```

---

### Step 3 — Pre-generate static data (first time only)

This generates all JSON data files used by the comparison, battle, and atlas pages:

```bash
# Still in backend/
python export_all.py
```

---

### Step 4 — Start the backend server

```bash
# Still in backend/
python main.py
```

Backend runs at **http://localhost:8000** and also serves the production build if `../build/` exists.

---

### Step 5 — Start the frontend (dev mode)

```bash
# Terminal 2 — from repo root
npm run dev
```

Frontend runs at **http://localhost:5173**  
Learn page runs at **http://localhost:5174**

---

### Production build (serves everything from one port)

```bash
# From repo root
npm run build

# Then just run the backend — it serves the built frontend too
cd backend
python main.py
```

Open **http://localhost:8000** — all pages served from one URL.

---

## 🌍 Public Access via ngrok

To share with others without a server:

```bash
# Install ngrok: https://ngrok.com/download
ngrok config add-authtoken YOUR_TOKEN

# Build first, then run backend, then tunnel
npm run build
cd backend && python main.py   # Terminal 1
ngrok http --domain=your-static-domain.ngrok-free.app 8000   # Terminal 2
```

Your app is now live at your ngrok URL.

---

## 👥 Team Members & Contributions

| Member | Contribution |
|--------|-------------|
| **Takshay Bansal** | Leading the ideation and strategy for project |
| **Saksham Gupta** | Developed BDH vs Transformer and No Code Studio |
| **Parate Aditya Nitin** | Proved and developed tech stack for Monosemanticity |
| **Kunchit Pujari** | Proved and developed tech stack for RoPE and ReLU functions |
| **Pourush Jalan** | Developed Frontend and training, inference & files |
| **Raghav K** | Developed Frontend and training, inference & files |
| **Dakshin Gautham** | Proved and developed tech stack for Hebbian learning & 3D brain topology |
| **Atharv Madan** | Developed learning course curriculum and Page architecture |

---

## ⚠️ Limitations & Future Scope

### Current Limitations
- The model is small (4-layer, 256-dim) — trained for interpretability demos, not SOTA performance
- The Learn page (`/learn`) runs as a separate Vite app and requires `npm run dev` or separate proxying in production
- Studio training runs on CPU by default; large jobs may be slow without a GPU
- Free ngrok URLs change on restart (use `--domain=` flag to pin the static domain)

### Future Scope
- Train a larger BDH variant and compare interpretability at scale
- Add real-time collaborative visualization (multiple users exploring the same model simultaneously)
- Integrate mechanistic interpretability techniques (activation patching, causal tracing)
- Deploy on a persistent VPS for always-on access
- Export neuron concept dictionaries as downloadable datasets
- Extend VS Battle to compare against GPT-2 and other open models
