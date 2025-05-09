#!/usr/bin/env python3
# inference_script.py  – runs inside SageMaker training container
# Folder structure in the container (/opt/ml/code):
#   /opt/ml/code/
#       inference_script.py
#       BIMBA-LLaVA-NeXT/
#       base_models/
#       video.mp4  (download target)

import subprocess, boto3, os, shutil, time
from pathlib import Path
from huggingface_hub import snapshot_download

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# ───────────────────────────── Paths ─────────────────────────────
ROOT            = Path(__file__).parent.resolve()              # /opt/ml/code
MODEL_DIR       = ROOT / "BIMBA-LLaVA-NeXT"                    # llava repo
CKPT_DIR        = MODEL_DIR / "checkpoints" / "BIMBA-LLaVA-Qwen2-7B"
BASE_MODEL_DIR  = ROOT / "base_models" / "LLaVA-Video-7B-Qwen2"
LOCAL_VIDEO     = MODEL_DIR / "video.mp4"
OUTPUT_JSON     = MODEL_DIR / "output.json"
S3_OUTPUT_KEY   = "bimba-output/output.json"

# ─────────────────────────── Download video ─────────────────────
video_s3 = os.environ["VIDEO_S3"]
bucket, key = video_s3.removeprefix("s3://").split("/", 1)

print("[INFO] Starting inference script.")
print(f"[INFO] Input video S3 path: {video_s3}")
print(f"[INFO] Downloading video to: {LOCAL_VIDEO}")

boto3.client("s3").download_file(bucket, key, str(LOCAL_VIDEO))
print("[INFO] Video downloaded successfully.")

# ──────────────── Refresh checkpoints from HF Hub ───────────────
print(f"[INFO] Checkpoint directory: {CKPT_DIR}")
if CKPT_DIR.exists():
    print("[INFO] Removing existing checkpoints…")
    shutil.rmtree(CKPT_DIR)

print("[INFO] Downloading checkpoints…")
snapshot_download(
    repo_id="mmiemon/BIMBA-LLaVA-Qwen2-7B",
    local_dir=str(CKPT_DIR),
    local_dir_use_symlinks=False,
    ignore_patterns=[".gitattributes"],
)
print("[INFO] Downloading base model/tokenizer…")
snapshot_download(
    repo_id="lmms-lab/LLaVA-Video-7B-Qwen2",
    repo_type="model",
    local_dir=str(BASE_MODEL_DIR),
    local_dir_use_symlinks=False,
)

# ───────────────────────────── Inference ────────────────────────
os.environ["MODEL_BASE"] = str(BASE_MODEL_DIR)
print(f"[INFO] MODEL_BASE set to: {BASE_MODEL_DIR}")

print("[INFO] Running LLaVA inference…")
start = time.time()
subprocess.run(["python", "inference.py"], cwd=str(MODEL_DIR), check=True)
print(f"[INFO] Inference completed in {time.time() - start:.2f}s.")

# ────────────────────────── Upload result ───────────────────────
if OUTPUT_JSON.exists():
    print("[INFO] Uploading output to S3…")
    boto3.client("s3").upload_file(str(OUTPUT_JSON), bucket, S3_OUTPUT_KEY)
    print(f"[INFO] Output uploaded to s3://{bucket}/{S3_OUTPUT_KEY}")
else:
    print(f"[WARNING] Output file {OUTPUT_JSON} not found; skipping upload.")

print("[INFO] Script finished.")
