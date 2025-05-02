import os
import subprocess
import logging
import shutil
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints/BIMBA-LLaVA-Qwen2-7B")
ckpt_file = os.path.join(ckpt_dir, "non_lora_trainables.bin")

# Refresh checkpoint directory
if os.path.exists(ckpt_dir):
    logger.info(f"Removing existing checkpoint dir: {ckpt_dir}")
    shutil.rmtree(ckpt_dir)

os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
logger.info("Cloning checkpoint from Hugging Face...")
subprocess.run(
    f"git clone https://huggingface.co/mmiemon/BIMBA-LLaVA-Qwen2-7B {ckpt_dir}",
    shell=True,
    check=True
)
logger.info("Clone complete.")

# Inspect bytes
with open(ckpt_file, "rb") as f:
    header = f.read(1024)
logger.info(f"[DEBUG] First 1KB of file:\n{header.decode(errors='ignore')}")


# # Try loading using safetensors
# try:
#     weights = load_file(ckpt_file)
#     logger.info("[SUCCESS] Loaded weights using safetensors.")
#     logger.info(f"Loaded keys: {list(weights.keys())[:5]}...")
# except Exception as e:
#     logger.exception("[FAIL] Error loading with safetensors:")
#     exit(1)
