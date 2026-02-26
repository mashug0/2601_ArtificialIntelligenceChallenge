"""
BDH Studio Backend — No-Code Training Platform
Handles specialist training, merging, and fine-tuning via SSE streams.
"""

import os
import uuid
import json
import math
import time
import queue
import threading
import traceback
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam, SGD
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/studio", tags=["studio"])

# ─────────────────────────── Job State ────────────────────────────

@dataclass
class JobState:
    job_id: str
    status: str = "idle"          # idle | queued | training | merging | finetuning | done | error
    current_epoch: int = 0
    total_epochs: int = 0
    train_losses: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    specialist_paths: Dict[str, str] = field(default_factory=dict)   # name → checkpoint path
    merged_path: Optional[str] = None
    finetuned_path: Optional[str] = None
    merged_snapshot: Optional[dict] = None
    results: Dict[str, Any] = field(default_factory=dict)            # final accuracy table
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    _queue: Any = field(default_factory=queue.Queue, repr=False)

JOBS: Dict[str, JobState] = {}

# ─────────────────────────── Dataset Info ─────────────────────────

DATASET_INFO = {
    "cifar10": {
        "name": "CIFAR-10",
        "n_classes": 10,
        "n_train": 50000,
        "n_test": 10000,
        "img_size": 32,
        "classes": ["Plane","Car","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"],
        "in_channels": 3,
    },
    "cifar100": {
        "name": "CIFAR-100",
        "n_classes": 100,
        "n_train": 50000,
        "n_test": 10000,
        "img_size": 32,
        "classes": [f"class_{i}" for i in range(100)],
        "in_channels": 3,
    },
}

# ─────────────────────────── Pydantic Models ──────────────────────

class ArchConfig(BaseModel):
    n_layer: int = 4              # Reduced from 6 for smaller model
    n_embd: int = 128             # Reduced from 192
    n_head: int = 4               # Reduced from 6
    mlp_internal_dim_multiplier: int = 16  # Reduced from 32 (halved)
    patch_size: int = 4
    top_k_fraction: float = 0.15
    dropout: float = 0.1
    use_rope: bool = True

class TrainConfig(BaseModel):
    epochs: int = 10              # Reduced from 50 for faster training
    batch_size: int = 32          # Reduced from 128 to prevent OOM
    learning_rate: float = 1e-4
    warmup_steps: int = 200       # Reduced proportionally
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    optimizer: str = "adamw"
    lr_schedule: str = "cosine"
    validation_split: float = 0.2
    aug_random_crop: bool = True
    aug_horizontal_flip: bool = True
    aug_color_jitter: bool = False
    aug_mixup: bool = False
    max_samples: int = 2000       # Use subset for quick PoC (2000 images instead of 50k)
    graph_update_every_n_batches: int = 10  # 0 = epoch-end only

class SpecialistSpec(BaseModel):
    name: str
    target_classes: List[int]
    arch: ArchConfig = ArchConfig()
    train: TrainConfig = TrainConfig()

class TrainRequest(BaseModel):
    dataset: str = "cifar10"
    specialists: List[SpecialistSpec]

class MergeRequest(BaseModel):
    job_id: str
    specialist_names: List[str]   # which specialists to merge
    finetune_after: bool = False
    finetune_epochs: int = 5      # Reduced from 15 for faster training
    finetune_lr: float = 5e-5

class FinetuneRequest(BaseModel):
    job_id: str
    epochs: int = 5               # Reduced from 15 for faster training
    learning_rate: float = 5e-5
    batch_size: int = 32          # Reduced from 64
    dataset: str = "cifar10"
    graph_update_every_n_batches: int = 10

# ─────────────────────────── Helpers ──────────────────────────────

def _push(job: JobState, msg: str):
    """Push a log line to the job queue and append to logs list."""
    job.logs.append(msg)
    if len(job.logs) > 200:
        job.logs = job.logs[-200:]
    job._queue.put({"type": "log", "message": msg})

def _push_progress(job: JobState, epoch: int, total: int, loss: float, val_acc: float, eta: float, specialist_name: Optional[str] = None):
    job.current_epoch = epoch
    job.total_epochs = total
    job.train_losses.append(loss)
    job.val_accs.append(val_acc)
    payload = {
        "type": "progress",
        "epoch": epoch,
        "total": total,
        "loss": round(loss, 4),
        "val_acc": round(val_acc, 2),
        "eta": round(eta, 1),
        "train_losses": job.train_losses[-50:],
        "val_accs": job.val_accs[-50:],
    }
    if specialist_name is not None:
        payload["specialist_name"] = specialist_name
    job._queue.put(payload)

def _push_done(job: JobState, data: dict = {}):
    job.status = "done"
    job._queue.put({"type": "done", **data})

def _push_error(job: JobState, err: str):
    job.status = "error"
    job.error = err
    job._queue.put({"type": "error", "message": err})


def _cosine_schedule(optimizer, warmup, total, last=-1):
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last)


def _get_param_count(arch: ArchConfig, num_classes: int) -> int:
    """Approximate param count for VisionBDH."""
    n = arch.n_embd
    heads = arch.n_head
    N = arch.mlp_internal_dim_multiplier * n // heads
    img_size = 32
    patch_size = arch.patch_size
    n_patches = (img_size // patch_size) ** 2
    patch_dim = 3 * patch_size * patch_size
    # patch embed + pos embed
    params = patch_dim * n + n_patches * n
    # per layer: enc + enc_v + dec + norms
    per_layer = 2 * (n * heads * N) + (heads * N * n) + 2 * n
    params += arch.n_layer * per_layer
    # head
    params += n * num_classes + num_classes
    return params


# ─────────────────────────── VisionBDH (inline) ───────────────────
# Inline implementation so no import dependency issues.

import dataclasses

@dataclasses.dataclass
class _BDHCfg:
    n_layer: int = 6
    n_embd: int = 192
    n_head: int = 6
    mlp_internal_dim_multiplier: int = 32
    vocab_size: int = 256
    dropout: float = 0.1
    max_seq_len: int = 128
    top_k_fraction: float = 0.15
    rope_theta: float = 10000.0

    @property
    def n_neurons_per_head(self):
        return self.mlp_internal_dim_multiplier * self.n_embd // self.n_head


def _get_freqs(D, theta, dtype):
    def quantize(t, q=2): return (t / q).floor() * q
    return 1.0 / (theta ** (quantize(torch.arange(0, D, 1, dtype=dtype)) / D)) / (2 * math.pi)


class _SparseAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.N = cfg.n_neurons_per_head
        self.register_buffer('freqs', _get_freqs(self.N, cfg.rope_theta, torch.float32).view(1,1,1,self.N))

    def rope(self, v):
        B, H, T, N = v.shape
        r = torch.arange(T, device=v.device, dtype=torch.float32).view(1,1,-1,1) * self.freqs
        c = torch.cos((r % 1) * 2 * math.pi).to(v.dtype)
        s = torch.sin((r % 1) * 2 * math.pi).to(v.dtype)
        v_rot = torch.stack((-v[...,1::2], v[...,::2]), dim=-1).view(B, H, T, N)
        return v * c + v_rot * s

    def forward(self, Q, K, V):
        B, T, _ = Q.size()
        q = self.rope(Q.view(B, T, self.n_head, self.N).transpose(1,2))
        k = self.rope(K.view(B, T, self.n_head, self.N).transpose(1,2))
        v = V.view(B, T, self.n_head, self.N).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.N))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1,2).contiguous().view(B, T, self.n_head * self.N)
        return out


class VisionBDH(nn.Module):
    """Vision BDH — patch embed + BDH core."""

    def __init__(self, arch: ArchConfig, img_size: int = 32, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.cfg = _BDHCfg(
            n_layer=arch.n_layer, n_embd=arch.n_embd, n_head=arch.n_head,
            mlp_internal_dim_multiplier=arch.mlp_internal_dim_multiplier,
            dropout=arch.dropout, top_k_fraction=arch.top_k_fraction,
        )
        cfg = self.cfg
        D, nh, N = cfg.n_embd, cfg.n_head, cfg.n_neurons_per_head

        n_patches = (img_size // arch.patch_size) ** 2
        patch_dim = in_channels * arch.patch_size * arch.patch_size

        self.patch_embed = nn.Conv2d(in_channels, D, kernel_size=arch.patch_size, stride=arch.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, D))
        self.ln_in = nn.LayerNorm(D)

        # Shared BDH weights (same across all layers — recurrent style)
        self.encoder   = nn.Parameter(torch.randn(nh, D, N) * 0.02)
        self.encoder_v = nn.Parameter(torch.randn(nh, D, N) * 0.02)
        self.decoder_weight = nn.Parameter(torch.randn(nh, N, D) * 0.02)
        self.decoder_bias   = nn.Parameter(torch.zeros(D))

        self.latent_norms_q = nn.ModuleList([nn.LayerNorm(nh * N) for _ in range(cfg.n_layer)])
        self.latent_norms_v = nn.ModuleList([nn.LayerNorm(nh * N) for _ in range(cfg.n_layer)])
        self.ln_out = nn.LayerNorm(D)
        self.drop = nn.Dropout(cfg.dropout)
        self.attn = _SparseAttn(cfg)

        self.head = nn.Linear(D, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def apply_kwta(self, x, fraction):
        k = max(1, int(x.shape[-1] * fraction))
        topk_vals, _ = torch.topk(x, k, dim=-1)
        threshold = topk_vals[..., -1:].detach()
        return x * (x >= threshold).float()

    def forward(self, x):
        cfg = self.cfg
        D, nh, N = cfg.n_embd, cfg.n_head, cfg.n_neurons_per_head

        B = x.size(0)
        # Patch embed
        x = self.patch_embed(x)           # B, D, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, T, D
        x = x + self.pos_embed
        curr = self.ln_in(x)

        W_enc   = self.encoder.permute(1,0,2).reshape(D, nh*N)
        W_enc_v = self.encoder_v.permute(1,0,2).reshape(D, nh*N)
        W_dec   = self.decoder_weight.reshape(nh*N, D)

        for i in range(cfg.n_layer):
            residual = curr
            q = self.apply_kwta(F.relu(self.latent_norms_q[i](curr @ W_enc)), cfg.top_k_fraction)
            v = self.apply_kwta(F.relu(self.latent_norms_v[i](curr @ W_enc_v)), cfg.top_k_fraction)
            y = self.attn(q, q, v) @ W_dec + self.decoder_bias
            curr = residual + self.drop(self.ln_out(y))

        # Global average pool → classify
        out = curr.mean(dim=1)
        return self.head(out)


# ─────────────────────────── Graph Snapshot ───────────────────────

def _extract_graph_snapshot(model, top_k_neurons: int = 16) -> dict:
    """
    Extract a layer-structured BDH weight graph for 3D visualization.
    Nodes are grouped by layer (Z-axis) and colored by head — mirrors the 3D Brain page layout.
    - n_layer planes on Z-axis (like the brain page's layer force)
    - Each plane contains n_head × top_k_neurons nodes
    - Residual edges connect same neuron across adjacent layers
    - Cross-head edges within each layer connect neurons with shared encoder features
    """
    with torch.no_grad():
        enc = model.encoder.detach().cpu()          # (nh, D, N)
        dec = model.decoder_weight.detach().cpu()   # (nh, N, D)
        nh, D, N = enc.shape
        n_layers = len(model.latent_norms_q)
        nodes, links = [], []
        seen_links: set = set()

        # Select top-k neurons per head by encoder L2 norm (consistent across layers)
        top_per_head = []
        for h in range(nh):
            scores = enc[h].norm(dim=0)   # (N,)
            k = min(top_k_neurons, N)
            top_per_head.append(scores.topk(k).indices.tolist())

        for l in range(n_layers):
            ln_q = model.latent_norms_q[l].weight.detach().cpu()  # (nh*N,)

            for h in range(nh):
                for ni in top_per_head[h]:
                    node_id  = f"l{l}_h{h}_n{ni}"
                    enc_norm = enc[h, :, ni].norm().item()
                    dec_norm = dec[h, ni].norm().item()
                    ln_w     = float(ln_q[h * N + ni].abs())

                    nodes.append({
                        "id":        node_id,
                        "layer":     l,        # Z-axis separation
                        "head":      h,        # coloring within each layer plane
                        "neuron":    ni,
                        "enc_norm":  round(enc_norm, 4),
                        "dec_norm":  round(dec_norm, 4),
                        "ln_weight": round(ln_w, 4),
                        "group":     l,        # layer = group for Z-force
                    })

                    # ── Residual edges: same (h, ni) across adjacent layers ──
                    if l < n_layers - 1:
                        w = round(min(dec_norm * 0.6, 1.0), 4)
                        links.append({
                            "source": node_id,
                            "target": f"l{l+1}_h{h}_n{ni}",
                            "weight": w,
                            "rtype":  "residual",
                        })

                    # ── Cross-head attention edges within same layer ──
                    # Connect to the top-responding neuron in adjacent head
                    next_h = (h + 1) % nh
                    # Find which neuron in next_h shares the strongest encoder feature
                    col_h    = enc[h,      :, ni].abs()      # (D,)
                    col_next = enc[next_h, :, :].abs()       # (D, N)
                    shared   = (col_h.unsqueeze(1) * col_next).sum(dim=0)  # (N,)
                    best_ni  = shared.argmax().item()
                    if best_ni in top_per_head[next_h]:
                        w_cross = round(float(shared[best_ni]) / (D * 0.1 + 1e-6), 4)
                        w_cross = min(max(w_cross, 0.02), 0.8)
                        lkey = f"{node_id}→l{l}_h{next_h}_n{best_ni}"
                        if lkey not in seen_links:
                            seen_links.add(lkey)
                            links.append({
                                "source": node_id,
                                "target": f"l{l}_h{next_h}_n{best_ni}",
                                "weight": round(w_cross, 4),
                                "rtype":  "attention",
                            })

    return {"nodes": nodes, "links": links}


# ─────────────────────────── Training Core ────────────────────────

def _run_training(job: JobState, dataset_key: str, spec: SpecialistSpec, num_classes_total: int):
    """Train one specialist. Runs in a background thread."""
    try:
        from torchvision.datasets import CIFAR10, CIFAR100
        from torchvision import transforms
        from torch.utils.data import DataLoader, random_split

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        arch = spec.arch
        train_cfg = spec.train
        target_classes = spec.target_classes
        num_classes = len(target_classes)

        if num_classes < 2:
            _push_error(job, f"[{spec.name}] Error: need at least 2 classes to train a classifier, got {num_classes} ({target_classes}). Assign 2+ classes to this specialist before running.")
            return

        label_map = {old: new for new, old in enumerate(target_classes)}

        _push(job, f"[{spec.name}] Starting on {device} | classes: {target_classes} ({num_classes} classes)")
        # Reset progress curves so Live Stats shows only this specialist
        job.train_losses = []
        job.val_accs = []
        # Notify frontend immediately which specialist is active (so progress bar isn't "queued")
        job._queue.put({
            "type": "progress",
            "epoch": 0,
            "total": train_cfg.epochs,
            "loss": 0,
            "val_acc": 0,
            "eta": 0,
            "train_losses": [],
            "val_accs": [],
            "specialist_name": spec.name,
        })

        # Build model
        model = VisionBDH(arch, img_size=32, in_channels=3, num_classes=num_classes).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        _push(job, f"[{spec.name}] Model: {total_params/1e6:.2f}M params")

        # Optimizer
        if train_cfg.optimizer == "adamw":
            opt = AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
        elif train_cfg.optimizer == "adam":
            opt = Adam(model.parameters(), lr=train_cfg.learning_rate)
        else:
            opt = SGD(model.parameters(), lr=train_cfg.learning_rate, momentum=0.9, weight_decay=train_cfg.weight_decay)

        # Transforms
        aug_list = [transforms.RandomResizedCrop(32, scale=(0.8, 1.0))] if train_cfg.aug_random_crop else [transforms.Resize(32)]
        if train_cfg.aug_horizontal_flip:
            aug_list.append(transforms.RandomHorizontalFlip())
        if train_cfg.aug_color_jitter:
            aug_list.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.1))
        aug_list += [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        transform_train = transforms.Compose(aug_list)
        transform_test  = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        # Dataset
        data_root = "./data_studio"
        os.makedirs(data_root, exist_ok=True)
        DatasetClass = CIFAR10 if dataset_key == "cifar10" else CIFAR100
        cifar_cache = os.path.join(data_root, "cifar-10-batches-py" if dataset_key == "cifar10" else "cifar-100-python")
        if os.path.exists(cifar_cache):
            _push(job, f"[{spec.name}] Loading dataset from cache…")
        else:
            _push(job, f"[{spec.name}] First run: downloading dataset (~170MB, one-time only)…")
        full_train = DatasetClass(root=data_root, train=True,  download=True, transform=transform_train)
        full_test  = DatasetClass(root=data_root, train=False, download=True, transform=transform_test)
        _push(job, f"[{spec.name}] Dataset ready ({len(full_train)} train / {len(full_test)} test images)")

        # Filter to target classes
        train_idx = [i for i, lbl in enumerate(full_train.targets) if lbl in target_classes]
        test_idx  = [i for i, lbl in enumerate(full_test.targets)  if lbl in target_classes]

        import numpy as np
        full_train.data    = full_train.data[train_idx]
        full_train.targets = [label_map[full_train.targets[i]] for i in train_idx]
        full_test.data     = full_test.data[test_idx]
        full_test.targets  = [label_map[full_test.targets[i]]  for i in test_idx]

        # PoC subsetting: limit total training images
        if train_cfg.max_samples > 0 and len(full_train.data) > train_cfg.max_samples:
            import random as _random
            idxs = _random.sample(range(len(full_train.data)), train_cfg.max_samples)
            full_train.data    = full_train.data[idxs]
            full_train.targets = [full_train.targets[i] for i in idxs]
            _push(job, f"[{spec.name}] PoC mode: using {len(idxs)} samples (of {len(train_idx)} available)")

        val_size   = int(train_cfg.validation_split * len(full_train))
        train_size = len(full_train) - val_size
        train_ds, val_ds = random_split(full_train, [train_size, val_size])

        train_loader = DataLoader(train_ds,   batch_size=train_cfg.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,     batch_size=train_cfg.batch_size, shuffle=False, num_workers=0, pin_memory=False)
        test_loader  = DataLoader(full_test,  batch_size=train_cfg.batch_size, shuffle=False, num_workers=0, pin_memory=False)

        total_steps = train_cfg.epochs * len(train_loader)
        if train_cfg.lr_schedule == "cosine":
            scheduler = _cosine_schedule(opt, train_cfg.warmup_steps, total_steps)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, train_cfg.warmup_steps)))

        loss_fn = nn.CrossEntropyLoss()
        ckpt_dir = f"./studio_checkpoints/{job.job_id}/{spec.name}"
        os.makedirs(ckpt_dir, exist_ok=True)

        # Training loop
        graph_n = train_cfg.graph_update_every_n_batches
        for epoch in range(train_cfg.epochs):
            if job.status in ("error", "cancelled"):
                return
            t0 = time.time()
            model.train()
            total_loss = 0.0
            for batch_idx, (imgs, lbls) in enumerate(train_loader):
                if job.status in ("error", "cancelled"):
                    return
                imgs, lbls = imgs.to(device), lbls.to(device)
                opt.zero_grad()
                loss = loss_fn(model(imgs), lbls)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                opt.step()
                scheduler.step()
                total_loss += loss.item()
                # Intra-epoch graph snapshot
                if graph_n > 0 and (batch_idx + 1) % graph_n == 0:
                    snapshot = _extract_graph_snapshot(model)
                    job._queue.put({"type": "graph", "snapshot": snapshot,
                                    "epoch": epoch, "batch": batch_idx + 1,
                                    "loss": round(total_loss / (batch_idx + 1), 4), "specialist_name": spec.name})

            avg_loss = total_loss / len(train_loader)
            epoch_time = time.time() - t0

            # Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    preds = model(imgs).argmax(dim=1)
                    total += lbls.size(0)
                    correct += (preds == lbls).sum().item()
            val_acc = 100 * correct / total

            eta = epoch_time * (train_cfg.epochs - epoch - 1)
            _push_progress(job, epoch + 1, train_cfg.epochs, avg_loss, val_acc, eta, specialist_name=spec.name)
            _push(job, f"[{spec.name}] Epoch {epoch+1}/{train_cfg.epochs} | loss={avg_loss:.4f} val={val_acc:.2f}% eta={eta:.0f}s")
            # Always send one graph snapshot at end of each epoch
            snapshot = _extract_graph_snapshot(model)
            job._queue.put({"type": "graph", "snapshot": snapshot,
                            "epoch": epoch + 1, "batch": len(train_loader),
                            "loss": round(avg_loss, 4), "val_acc": round(val_acc, 2), "specialist_name": spec.name})

        # Final test accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = model(imgs).argmax(dim=1)
                total += lbls.size(0)
                correct += (preds == lbls).sum().item()
        test_acc = 100 * correct / total
        _push(job, f"[{spec.name}] ✓ Test accuracy: {test_acc:.2f}%")

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, "final.pth")
        torch.save({
            "model_state_dict": model.state_dict(),
            "arch": arch.dict(),
            "target_classes": target_classes,
            "num_classes": num_classes,
            "test_acc": test_acc,
            "val_acc": val_acc,
        }, ckpt_path)
        job.specialist_paths[spec.name] = ckpt_path
        job.results[spec.name] = {"test_acc": test_acc, "val_acc": val_acc, "path": ckpt_path}
        _push(job, f"[{spec.name}] Saved → {ckpt_path}")

    except Exception as e:
        tb = traceback.format_exc()
        _push_error(job, f"[{spec.name}] Error: {str(e)}\n{tb}")


def _run_all_specialists(job: JobState, req: TrainRequest):
    """Run all specialists sequentially, then signal done."""
    try:
        job.status = "training"
        info = DATASET_INFO.get(req.dataset, DATASET_INFO["cifar10"])
        for spec in req.specialists:
            if job.status in ("error", "cancelled"):
                return
            _push(job, f"━━━ Starting Specialist: {spec.name} ━━━")
            _run_training(job, req.dataset, spec, info["n_classes"])
        job.status = "done"
        _push_done(job, {"results": job.results, "specialist_paths": job.specialist_paths})
    except Exception as e:
        _push_error(job, str(e))


# ─────────────────────────── Merge Core ───────────────────────────

def _merge_state_dicts(state_a: dict, state_b: dict) -> dict:
    """
    Merge two VisionBDH state dicts by concatenating along the neuron dimension.
    Based on paper Section 7.1: concat along N (internal neuron dim), keep I/O same.
    """
    merged = {}
    for key in state_a:
        sa, sb = state_a[key], state_b[key]

        if "head.weight" in key:
            # Specialists have different local class spaces (e.g. [5,D] each).
            # Stack along dim=0 to get [10, D] — fine-tuning will re-learn this head anyway.
            if sa.shape == sb.shape:
                merged[key] = torch.cat([sa, sb], dim=0)   # [2*num_classes, D]
            else:
                merged[key] = sa   # fallback: keep first specialist's head
        elif "head.bias" in key:
            if sa.shape == sb.shape:
                merged[key] = torch.cat([sa, sb], dim=0)   # [2*num_classes]
            else:
                merged[key] = sa
        elif "encoder" in key and "encoder_v" not in key:
            # [n_head, D, N] → concat on N (dim=2), or [D, N] concat dim=1
            concat_dim = 2 if sa.dim() == 3 else (1 if sa.dim() == 2 else -1)
            merged[key] = torch.cat([sa, sb], dim=concat_dim)
        elif "encoder_v" in key:
            concat_dim = 2 if sa.dim() == 3 else (1 if sa.dim() == 2 else -1)
            merged[key] = torch.cat([sa, sb], dim=concat_dim)
        elif "decoder_weight" in key:
            # [n_head, N, D] → concat on N (dim=1)
            concat_dim = 1 if sa.dim() == 3 else 0
            merged[key] = torch.cat([sa, sb], dim=concat_dim)
        elif "decoder_bias" in key:
            merged[key] = (sa + sb) / 2.0
        elif "latent_norm" in key:
            # LayerNorm weight/bias: [n_head * N] → concat dim=0
            merged[key] = torch.cat([sa, sb], dim=0)
        elif "patch_embed" in key or "pos_embed" in key:
            merged[key] = (sa + sb) / 2.0
        elif "ln_in" in key or "ln_out" in key:
            # LayerNorm on D-dim — stays same size, average
            merged[key] = (sa + sb) / 2.0
        elif "freqs" in key:
            merged[key] = torch.cat([sa, sb], dim=-1)
        else:
            merged[key] = (sa + sb) / 2.0
    return merged


def _run_merge(job: JobState, req: MergeRequest):
    try:
        job.status = "merging"
        _push(job, "━━━ Starting Model Merge ━━━")

        # Load specialists
        states = []
        archs = []
        for name in req.specialist_names:
            path = job.specialist_paths.get(name)
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint for {name} not found at {path}")
            ckpt = torch.load(path, map_location="cpu")
            states.append(ckpt["model_state_dict"])
            archs.append(ckpt["arch"])
            _push(job, f"  Loaded {name}: test_acc={ckpt.get('test_acc', '?'):.2f}%")

        # Merge pairwise
        merged_state = states[0]
        for i in range(1, len(states)):
            _push(job, f"  Merging specialist 0..{i-1} with specialist {i}...")
            merged_state = _merge_state_dicts(merged_state, states[i])

        # Build merged arch config (double the mlp multiplier)
        base_arch = ArchConfig(**archs[0])
        merged_arch = ArchConfig(
            n_layer=base_arch.n_layer,
            n_embd=base_arch.n_embd,
            n_head=base_arch.n_head,
            mlp_internal_dim_multiplier=base_arch.mlp_internal_dim_multiplier * len(states),
            patch_size=base_arch.patch_size,
            top_k_fraction=base_arch.top_k_fraction,
            dropout=base_arch.dropout,
        )

        # Collect all classes from all specialists
        all_classes = []
        for name in req.specialist_names:
            path = job.specialist_paths[name]
            ckpt = torch.load(path, map_location="cpu")
            all_classes.extend(ckpt.get("target_classes", []))
        num_merged_classes = len(set(all_classes))

        merged_dir = f"./studio_checkpoints/{job.job_id}/merged"
        os.makedirs(merged_dir, exist_ok=True)
        merged_path = os.path.join(merged_dir, "merged.pth")

        torch.save({
            "model_state_dict": merged_state,
            "arch": merged_arch.dict(),
            "all_classes": sorted(set(all_classes)),
            "num_classes": num_merged_classes,
            "merged_from": req.specialist_names,
        }, merged_path)

        job.merged_path = merged_path
        params_approx = _get_param_count(merged_arch, num_merged_classes)
        _push(job, f"  ✓ Merged model saved → {merged_path}")
        _push(job, f"  ✓ Approx params: {params_approx/1e6:.2f}M (doubled internal N)")
        job.results["merged"] = {"path": merged_path, "arch": merged_arch.dict(), "params": params_approx}

        # Extract graph snapshot for 3D visualisation of merged architecture (more neurons = num_merged specialists)
        try:
            merged_model = VisionBDH(merged_arch, img_size=32, in_channels=3, num_classes=num_merged_classes).to("cpu")
            merged_model.load_state_dict(merged_state, strict=False)
            merged_model.eval()
            top_k = 16 * len(states)  # scale neuron count in graph by number of merged models
            job.merged_snapshot = _extract_graph_snapshot(merged_model, top_k_neurons=min(top_k, merged_arch.mlp_internal_dim_multiplier * merged_arch.n_embd // merged_arch.n_head))
            _push(job, f"  ✓ Graph snapshot extracted ({len(job.merged_snapshot['nodes'])} neurons)")
        except Exception as snap_err:
            _push(job, f"  ⚠ Could not extract merged graph snapshot: {snap_err}")
            job.merged_snapshot = None

        if req.finetune_after:
            ft_req = FinetuneRequest(
                job_id=job.job_id, epochs=req.finetune_epochs,
                learning_rate=req.finetune_lr, batch_size=64
            )
            _run_finetune(job, ft_req)
        else:
            job.status = "done"
            _push_done(job, {
                "merged_path": merged_path,
                "results": job.results,
                "merged_snapshot": job.merged_snapshot,
            })

    except Exception as e:
        tb = traceback.format_exc()
        _push_error(job, f"Merge error: {str(e)}\n{tb}")


def _run_finetune(job: JobState, req: FinetuneRequest):
    try:
        job.status = "finetuning"
        _push(job, "━━━ Starting Fine-tune on Full Dataset ━━━")

        if not job.merged_path or not os.path.exists(job.merged_path):
            raise FileNotFoundError("No merged model found. Run merge first.")

        ckpt = torch.load(job.merged_path, map_location="cpu")
        arch = ArchConfig(**ckpt["arch"])
        all_classes = ckpt.get("all_classes", list(range(10)))
        num_classes  = ckpt.get("num_classes", 10)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = VisionBDH(arch, img_size=32, in_channels=3, num_classes=num_classes).to(device)

        # Strip head weights from checkpoint — they may have wrong output size
        # (specialist head = 5 classes, merged model needs 10 classes).
        # Load backbone weights only, then let the freshly-initialized head train.
        state = {k: v for k, v in ckpt["model_state_dict"].items()
                 if not k.startswith("head.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        head_keys = [k for k in ckpt["model_state_dict"] if k.startswith("head.")]
        _push(job, f"  Backbone loaded. Head re-initialized for {num_classes} classes.")
        if missing:
            non_head_missing = [k for k in missing if not k.startswith("head.")]
            if non_head_missing:
                _push(job, f"  ⚠ Missing backbone keys: {non_head_missing[:5]}")

        _push(job, f"  Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        from torchvision.datasets import CIFAR10, CIFAR100
        from torchvision import transforms
        from torch.utils.data import DataLoader, random_split

        label_map = {old: new for new, old in enumerate(sorted(all_classes))}
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

        data_root = "./data_studio"
        DatasetClass = CIFAR10 if req.dataset == "cifar10" else CIFAR100
        full_train = DatasetClass(root=data_root, train=True,  download=True, transform=transform_train)
        full_test  = DatasetClass(root=data_root, train=False, download=True, transform=transform_test)

        # Filter to only the classes we trained on
        train_idx = [i for i, lbl in enumerate(full_train.targets) if lbl in all_classes]
        test_idx  = [i for i, lbl in enumerate(full_test.targets)  if lbl in all_classes]
        import numpy as np
        full_train.data    = full_train.data[train_idx]
        full_train.targets = [label_map[full_train.targets[i]] for i in train_idx]
        full_test.data     = full_test.data[test_idx]
        full_test.targets  = [label_map[full_test.targets[i]]  for i in test_idx]

        val_size   = int(0.2 * len(full_train))
        train_size = len(full_train) - val_size
        train_ds, val_ds = random_split(full_train, [train_size, val_size])

        train_loader = DataLoader(train_ds,   batch_size=req.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,     batch_size=req.batch_size, shuffle=False, num_workers=0, pin_memory=False)
        test_loader  = DataLoader(full_test,  batch_size=req.batch_size, shuffle=False, num_workers=0, pin_memory=False)

        opt = AdamW(model.parameters(), lr=req.learning_rate, weight_decay=0.01)
        total_steps = req.epochs * len(train_loader)
        scheduler = _cosine_schedule(opt, min(500, total_steps // 10), total_steps)
        loss_fn = nn.CrossEntropyLoss()

        best_val = 0.0
        ft_dir = f"./studio_checkpoints/{job.job_id}/finetuned"
        os.makedirs(ft_dir, exist_ok=True)

        graph_n = req.graph_update_every_n_batches
        # Use merged-scale top_k so graph shows same neuron count as merge (not specialist-scale)
        _, _, N_enc = model.encoder.shape
        top_k_ft = min(64, N_enc)

        for epoch in range(req.epochs):
            if job.status == "error":
                return
            t0 = time.time()
            model.train()
            total_loss = 0.0
            for batch_idx, (imgs, lbls) in enumerate(train_loader):
                imgs, lbls = imgs.to(device), lbls.to(device)
                opt.zero_grad()
                loss = loss_fn(model(imgs), lbls)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                total_loss += loss.item()

                if graph_n > 0 and (batch_idx + 1) % graph_n == 0:
                    snapshot = _extract_graph_snapshot(model, top_k_neurons=top_k_ft)
                    job._queue.put({"type": "graph", "snapshot": snapshot,
                                    "epoch": epoch + 1, "batch": batch_idx + 1,
                                    "loss": round(total_loss / (batch_idx + 1), 4)})

            avg_loss = total_loss / len(train_loader)
            epoch_time = time.time() - t0

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    preds = model(imgs).argmax(dim=1)
                    total += lbls.size(0)
                    correct += (preds == lbls).sum().item()
            val_acc = 100 * correct / total
            eta = epoch_time * (req.epochs - epoch - 1)

            _push_progress(job, epoch + 1, req.epochs, avg_loss, val_acc, eta)
            _push(job, f"[finetune] Epoch {epoch+1}/{req.epochs} | loss={avg_loss:.4f} val={val_acc:.2f}%")

            # Epoch-end graph snapshot (merged-scale neurons)
            snapshot = _extract_graph_snapshot(model, top_k_neurons=top_k_ft)
            job._queue.put({"type": "graph", "snapshot": snapshot,
                            "epoch": epoch + 1, "batch": len(train_loader),
                            "loss": round(avg_loss, 4), "val_acc": round(val_acc, 2)})

            if val_acc > best_val:
                best_val = val_acc
                torch.save({"model_state_dict": model.state_dict(), "arch": arch.dict(),
                            "val_acc": val_acc, "all_classes": all_classes}, os.path.join(ft_dir, "best.pth"))

        # Final test
        best_ckpt = torch.load(os.path.join(ft_dir, "best.pth"), map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = model(imgs).argmax(dim=1)
                total += lbls.size(0)
                correct += (preds == lbls).sum().item()
        test_acc = 100 * correct / total

        ft_path = os.path.join(ft_dir, "finetuned_final.pth")
        torch.save(best_ckpt, ft_path)
        job.finetuned_path = ft_path
        job.results["finetuned"] = {"test_acc": test_acc, "val_acc": best_val, "path": ft_path}

        _push(job, f"[finetune] ✓ Test accuracy: {test_acc:.2f}%")
        job.status = "done"
        _push_done(job, {"finetuned_path": ft_path, "test_acc": test_acc, "results": job.results})

    except Exception as e:
        tb = traceback.format_exc()
        _push_error(job, f"Finetune error: {str(e)}\n{tb}")


# ─────────────────────────── Inference ───────────────────────────

CIFAR10_CLASSES  = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
CIFAR100_CLASSES = [str(i) for i in range(100)]  # fallback; CIFAR-100 has fine labels

class InferRequest(BaseModel):
    checkpoint_path: str   # absolute or relative path to .pth checkpoint
    image_b64: str         # base64-encoded image (PNG/JPEG)
    dataset: str = "cifar10"

@router.post("/infer")
async def run_inference(req: InferRequest):
    """
    Run a forward pass on a single uploaded image through a trained checkpoint.
    Returns class probabilities + a graph snapshot with per-neuron activation levels.
    """
    import base64, io
    from PIL import Image
    from torchvision import transforms

    # ── Load checkpoint ──────────────────────────────────────────
    ckpt_path = os.path.normpath(req.checkpoint_path.strip())
    if not os.path.exists(ckpt_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)

    arch_dict = ckpt.get("arch", {})
    # Specialist/merged checkpoints have num_classes + target_classes; finetuned has all_classes only
    all_classes = ckpt.get("all_classes")
    if all_classes is not None:
        num_classes = len(all_classes)
        target_classes = sorted(all_classes)
    else:
        target_classes = ckpt.get("target_classes", list(range(ckpt.get("num_classes", 10))))
        num_classes = ckpt.get("num_classes", len(target_classes))

    arch = ArchConfig(**arch_dict) if arch_dict else ArchConfig()
    model = VisionBDH(arch, img_size=32, in_channels=3, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Decode & preprocess image ─────────────────────────────────
    img_bytes = base64.b64decode(req.image_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    x = transform(img).unsqueeze(0).to(device)  # [1, 3, 32, 32]

    # ── Forward pass with activation hooks ───────────────────────
    layer_activations: list = []   # per BDH layer: (nh, N) tensor of mean abs activation

    hooks = []
    def make_hook(layer_idx):
        def hook(module, inp, out):
            # out is the post-attention output before residual — grab the hidden state
            # We hook after the encoder matmul; capture encoder output
            pass
        return hook

    # Better approach: hook the latent_norms_q (one per layer, shape [n_head*N])
    # to capture post-norm activations as a proxy for neuron firing
    norm_outputs = []
    def norm_hook(module, inp, out):
        norm_outputs.append(out.detach().cpu())  # [B, T, n_head*N] or [n_head*N]

    for ln in model.latent_norms_q:
        hooks.append(ln.register_forward_hook(norm_hook))

    with torch.no_grad():
        logits = model(x)          # [1, num_classes]
        probs  = F.softmax(logits[0], dim=-1).cpu().tolist()

    for h in hooks:
        h.remove()

    # ── Build activation map: node_id → activation (0-1 normalized) ──
    # norm_outputs[l] shape: [1, n_patches, n_head*N] → mean over patches → [n_head*N]
    activation_map: dict = {}
    nh = arch.n_head
    n_layers = arch.n_layer

    for l_idx, act_tensor in enumerate(norm_outputs[:n_layers]):
        # act_tensor: [1, T, nh*N] — mean pool over spatial tokens
        act = act_tensor[0].abs().mean(dim=0)  # [nh*N]
        N_dim = act.shape[0] // nh if nh > 0 else act.shape[0]
        act_max = act.max().item() + 1e-6
        for h in range(nh):
            for ni in range(N_dim):
                node_id = f"l{l_idx}_h{h}_n{ni}"
                raw = act[h * N_dim + ni].item()
                activation_map[node_id] = round(raw / act_max, 4)

    # ── Graph snapshot with activation overlay ────────────────────
    snapshot = _extract_graph_snapshot(model)
    for node in snapshot["nodes"]:
        node["activation"] = activation_map.get(node["id"], 0.0)

    # ── Class names ───────────────────────────────────────────────
    if req.dataset == "cifar10":
        all_names = CIFAR10_CLASSES
    else:
        all_names = CIFAR100_CLASSES
    class_names = [all_names[c] if c < len(all_names) else str(c) for c in target_classes]

    predicted = int(torch.tensor(probs).argmax().item())

    return {
        "class_probs": probs,
        "class_names": class_names,
        "predicted_class": predicted,
        "graph_snapshot": snapshot,
    }


# ─────────────────────────── API Routes ───────────────────────────

@router.get("/datasets")
async def list_datasets():
    return {"datasets": DATASET_INFO}


@router.get("/param-count")
async def estimate_params(n_layer: int = 6, n_embd: int = 192, n_head: int = 6,
                           mlp_internal_dim_multiplier: int = 32, num_classes: int = 10):
    arch = ArchConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head,
                      mlp_internal_dim_multiplier=mlp_internal_dim_multiplier)
    count = _get_param_count(arch, num_classes)
    return {"params": count, "params_m": round(count / 1e6, 2)}


@router.post("/train")
async def start_training(req: TrainRequest):
    job_id = str(uuid.uuid4())[:8]
    job = JobState(job_id=job_id, status="queued")
    JOBS[job_id] = job

    t = threading.Thread(target=_run_all_specialists, args=(job, req), daemon=True)
    t.start()

    return {"job_id": job_id, "status": "queued", "message": f"Training {len(req.specialists)} specialist(s)"}


@router.post("/cancel/{job_id}")
async def cancel_training(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in ("training", "merging", "finetuning", "queued"):
        job.status = "cancelled"
        job._queue.put({"type": "cancelled", "message": "Training cancelled by user"})
    return {"job_id": job_id, "status": job.status}


@router.get("/stream/{job_id}")
async def stream_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]

    def event_generator():
        while True:
            try:
                msg = job._queue.get(timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                # Send heartbeat
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                if job.status in ("done", "error"):
                    break

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/job/{job_id}")
async def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    d = asdict(job)
    d.pop("_queue", None)
    return d


@router.get("/jobs")
async def list_jobs():
    result = []
    for job_id, job in JOBS.items():
        result.append({
            "job_id": job_id,
            "status": job.status,
            "current_epoch": job.current_epoch,
            "total_epochs": job.total_epochs,
            "specialist_paths": list(job.specialist_paths.keys()),
            "results": job.results,
        })
    return {"jobs": result}


@router.post("/merge")
async def merge_models(req: MergeRequest):
    if req.job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[req.job_id]

    if not job.specialist_paths:
        raise HTTPException(status_code=400, detail="No trained specialists found. Train first.")

    missing = [n for n in req.specialist_names if n not in job.specialist_paths]
    if missing:
        raise HTTPException(status_code=400, detail=f"Specialists not trained yet: {missing}")

    t = threading.Thread(target=_run_merge, args=(job, req), daemon=True)
    t.start()
    return {"job_id": req.job_id, "status": "merging"}


@router.post("/finetune")
async def finetune_model(req: FinetuneRequest):
    if req.job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[req.job_id]

    if not job.merged_path:
        raise HTTPException(status_code=400, detail="No merged model found. Run merge first.")

    t = threading.Thread(target=_run_finetune, args=(job, req), daemon=True)
    t.start()
    return {"job_id": req.job_id, "status": "finetuning"}


@router.get("/checkpoints/{job_id}")
async def list_checkpoints(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    return {
        "specialists": job.specialist_paths,
        "merged": job.merged_path,
        "finetuned": job.finetuned_path,
        "results": job.results,
        "merged_snapshot": job.merged_snapshot,
    }
