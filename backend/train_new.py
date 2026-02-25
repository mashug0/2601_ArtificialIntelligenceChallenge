"""
BDH Final Showcase: The "Goldilocks" Run
Data: War & Peace + Sherlock Holmes (~4MB)
"""

import os
import torch
import requests
from tqdm import tqdm
from model_new import BDH, BDHConfig
from custom_tokenizer import get_custom_tokenizer

# --- CONFIG ---
CHECKPOINT_PATH = "checkpoint_final.pt"
TOTAL_STEPS = 1500
GEN_EVERY = 500

# 1. DATA: Contrast Mix (War vs Mystery)
DATA_URLS = [
    "https://raw.githubusercontent.com/mmcky/nyu-econ-370/master/notebooks/data/book-war-and-peace.txt",
    "https://www.gutenberg.org/files/1661/1661-0.txt" # Sherlock Holmes
]

print("📥 Loading data...")
text = ""
for url in DATA_URLS:
    try:
        r = requests.get(url)
        if r.status_code == 200:
            c = r.text
            if "*** START" in c: c = c.split("*** START")[1]
            if "*** END" in c: c = c.split("*** END")[0]
            text += c + "\n"
    except: pass

print(f"Total Size: {len(text)/1024/1024:.2f} MB")

print("🔤 Tokenizer...")
try:
    tokenizer = get_custom_tokenizer(text, vocab_size=10000)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    vocab_size = tokenizer.n_vocab
except:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    vocab_size = 50257
    tokenizer = enc

train_data = data[:int(0.9*len(data))]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. CONFIG: The "Sweet Spot"
config = BDHConfig(
    n_layer=4, 
    n_embd=256, 
    n_head=4, 
    mlp_internal_dim_multiplier=8,
    vocab_size=vocab_size, 
    dropout=0.1,
    top_k_fraction=0.15 # 15% Sparsity
)

model = BDH(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

print(f"\n🔥 Training (Novels, k-WTA 15%)\n")

model.train()
pbar = tqdm(range(1, TOTAL_STEPS+1))

for step in pbar:
    ix = torch.randint(len(train_data) - 128 - 1, (32,))
    x = torch.stack([train_data[i:i+128] for i in ix]).to(device)
    y = torch.stack([train_data[i+1:i+129] for i in ix]).to(device)

    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss, diag = model(x, targets=y, return_diagnostics=(step%100==0))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 10 == 0:
        pbar.set_postfix({"Loss": f"{loss.item():.3f}"})

    # GENERATION & PROBING
    if step % GEN_EVERY == 0:
        model.eval()
        print(f"\n{'='*50}\nStep {step} | Loss {loss.item():.3f}")
        
        # 3. PROBES
        pairs = [
            ("soldier", "army"),
            ("eyes", "face"),
            ("door", "room"),
            ("General", "flower"),
            ("room", "sky"),
            ("gun", "kiss") 
        ]
        
        sims = model.semantic_probe(pairs, tokenizer)
        
        print(f"\n🧠 Semantic Separation:")
        for w1, w2 in pairs:
            s = sims.get(f"{w1}-{w2}", 0.0)
            high_group = ["soldier", "eyes", "door"]
            exp = "HIGH" if w1 in high_group else "LOW"
            passed = (exp == "HIGH" and s > 0.4) or (exp == "LOW" and s < 0.4)
            mark = "✓" if passed else "✗"
            print(f"  {w1:>8} - {w2:<8} : {s:.3f}  ({exp}) {mark}")

        # GEN
        ctx = torch.tensor(tokenizer.encode("The general said")).unsqueeze(0).to(device)
        out = model.generate(ctx, 50, top_k=40)
        print(f"\nGen: {tokenizer.decode(out[0].tolist())}\n{'='*50}")
        
        # [ADDED] SAVE CHECKPOINT PERIODICALLY
        torch.save({
            'model': model.state_dict(),
            'config': config,
            'optimizer': optimizer.state_dict()
        }, CHECKPOINT_PATH)
        print(f"💾 Checkpoint saved to {CHECKPOINT_PATH}")

        model.train()

print("Training Complete.")
torch.save({
    'model': model.state_dict(),
    'config': config,
    'optimizer': optimizer.state_dict()
}, CHECKPOINT_PATH)
print(f"✅ Final Model Saved to {CHECKPOINT_PATH}")