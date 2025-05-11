#!/usr/bin/env python3
"""
train_entrypoint.py – SageMaker wrapper for BIMBA training
"""

import subprocess, yaml, os, sys
from pathlib import Path

ROOT      = Path(__file__).parent.resolve()            # /opt/ml/code
REPO_DIR  = ROOT / "BIMBA-LLaVA-NeXT"                  # /opt/ml/code/BIMBA-LLaVA-NeXT/
EXP_FILE  = REPO_DIR / "scripts/video/train/exp.yaml"  # /opt/ml/code/BIMBA-LLaVA-NeXT/scripts/video/train/exp.yaml
TRAIN_CH  = Path("/opt/ml/input/data/training")
JSONL     = next(TRAIN_CH.glob("*.jsonl"))             # first *.jsonl
VIDEO_DIR = TRAIN_CH                                   # ─┐ use same channel
IMAGE_DIR = TRAIN_CH                                   # ─┘ for now

print(f"[INFO] Training JSONL: {JSONL}")

# ── 1. Make the repo importable  ──────────────────────────────
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", "."],
    cwd=str(REPO_DIR),
    check=True,
)

def ensure_flash_attn():
    # Pick ONE of the branches below ⬇
    try:
        # A) exact wheel – fastest
        wheel = ("https://github.com/Dao-AILab/flash-attention/releases/"
                 "download/v2.7.4.post1/"
                 "flash_attn-2.7.4.post1+cu121torch2.1cxx11abiFALSE-"
                 "cp310-cp310_linux_x86_64.whl")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", wheel],
                       check=True)
    except subprocess.CalledProcessError:
        # B) fall back to the last cu121-torch2.1 wheel series
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "flash-attn==2.4.*"], check=True)
    # C) if *both* fail, disable Flash-Attn entirely
    except Exception as e:
        print("[WARN] flash-attn wheel install failed → using SDPA:", e)
        os.environ["FLASH_ATTENTION_2"] = "0"

ensure_flash_attn()

# ── 2. Patch exp.yaml so LLaVA knows where the JSONL lives ────
with EXP_FILE.open() as f:
    cfg = yaml.safe_load(f)

cfg["datasets"][0]["json_path"] = str(JSONL)

with EXP_FILE.open("w") as f:
    yaml.safe_dump(cfg, f)

print(f"[INFO] Patched exp.yaml -> {cfg['datasets'][0]['json_path']}")

# ── 3. Patch the bash script placeholders on the fly ──────────
train_sh = REPO_DIR / "scripts/video/train/Train_BIMBA_LLaVA_Qwen2_7B.sh"
text = train_sh.read_text()
text = text.replace("data_path XXX",    f"data_path {JSONL}")        \
           .replace("image_folder XXX", f"image_folder {IMAGE_DIR}") \
           .replace("video_folder XXX", f"video_folder {VIDEO_DIR}")
train_sh.write_text(text)

# ── 4. Kick off training  ─────────────────────────────────────
print("[INFO] Starting BIMBA training ...")
os.environ["PYTHONPATH"] = f"{REPO_DIR}:{os.environ.get('PYTHONPATH','')}"
subprocess.run(["bash", str(train_sh)], cwd=str(REPO_DIR), check=True)
print("[INFO] Training finished.")
