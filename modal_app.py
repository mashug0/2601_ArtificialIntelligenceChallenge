"""
BDH — Modal Deployment
=====================
Setup (one-time):
    pip install modal
    modal setup              # authenticate with your Modal account

Build frontend first:
    npm run build            # generates build/ folder

Deploy (makes it live, laptop can be closed):
    modal deploy modal_app.py

Test locally before deploying:
    modal serve modal_app.py
"""

import modal
from pathlib import Path

ROOT = Path(__file__).parent

# ── Persistent volumes ────────────────────────────────────────────────────────
# Studio checkpoints persist across cold starts / redeployments
checkpoints_vol = modal.Volume.from_name("bdh-studio-checkpoints", create_if_missing=True)
# CIFAR dataset cache — avoids re-downloading on every cold start
data_vol = modal.Volume.from_name("bdh-data-studio", create_if_missing=True)

# ── Container image ───────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "numpy>=1.24.0",
    )
    .pip_install(
        "torch",
        "torchvision",   # required by studio.py (CIFAR10/CIFAR100 datasets)
        extra_index_url="https://download.pytorch.org/whl/cu121",  # CUDA 12.1 wheels
    )
    .pip_install(
        "pydantic>=2.6.0",
        "python-multipart>=0.0.6",
        "networkx>=3.0",
        "scikit-learn>=1.3.0",
        "tokenizers>=0.15.0",
        "tqdm>=4.65.0",
    )
    .add_local_dir(str(ROOT / "backend"), remote_path="/app/backend")
    .add_local_dir(str(ROOT / "build"),   remote_path="/app/build")    # npm run build output
    .add_local_dir(str(ROOT / "static"),  remote_path="/app/static")
)

# ── Modal app ─────────────────────────────────────────────────────────────────
app = modal.App("bdh-bdh", image=image)


@app.function(
    gpu="T4",           # cheapest Modal GPU (~$0.59/hr); swap to "A10G" for faster Studio training
    memory=4096,
    timeout=300,
    container_idle_timeout=120,   # scale to zero after 2 min idle
    # Mount volumes at clean /mnt/ paths to avoid conflicts with add_local_dir content.
    # Symlinks created at runtime redirect the backend to these mount points.
    volumes={
        "/mnt/studio_checkpoints": checkpoints_vol,
        "/mnt/data_studio": data_vol,
    },
    # Keep exactly 1 container — CRITICAL for Studio training.
    # The JOBS dict is in-memory; if Modal autoscales to 2+ containers,
    # the SSE stream request can hit a different container than where the
    # training thread is running, and the job won't be found.
    min_containers=1,
    max_containers=1,
)
@modal.asgi_app()
def fastapi_app():
    import sys, os
    sys.path.insert(0, "/app/backend")
    os.chdir("/app/backend")

    # Replace the local copies with symlinks to the persistent volume mount points.
    # This lets backend code use the same relative paths without modification.
    import shutil
    for src, dst in [
        ("/mnt/studio_checkpoints", "/app/backend/studio_checkpoints"),
        ("/mnt/data_studio",        "/app/backend/data_studio"),
    ]:
        if os.path.islink(dst):
            pass  # already linked (warm container restart)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)
            os.symlink(src, dst)
        else:
            os.symlink(src, dst)

    from main import app as fastapi_application
    return fastapi_application