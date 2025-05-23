from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
from pathlib import Path
import json
import os
warnings.filterwarnings("ignore")

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames,frame_time,video_time

# model_path = "checkpoints/BIMBA-LLaVA-Qwen2-7B"
# model_path = str(Path(__file__).resolve().parent / "BIMBA/BIMBA-LLaVA-NeXT/checkpoints/BIMBA-LLaVA-Qwen2-7B")
# model_path = str(Path(__file__).resolve().parent / "checkpoints/BIMBA-LLaVA-Qwen2-7B")

repo_root = Path(__file__).resolve().parent
model_path = str(repo_root / "checkpoints/BIMBA-LLaVA-Qwen2-7B")



# model_base = "lmms-lab/LLaVA-Video-7B-Qwen2"
# model_base = os.environ["MODEL_BASE"]
model_base = os.path.abspath(os.environ["MODEL_BASE"])
model_name = "llava_qwen_lora"


device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(
                                                    model_path = model_path, 
                                                    model_base = model_base, 
                                                    model_name = model_name, 
                                                    torch_dtype="bfloat16", 
                                                    device_map=device_map,
                                                    attn_implementation=None
                                                )

model.eval()


# video_path = "assets/example.mp4"
video_path = "video.mp4"
max_frames_num = 64
video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
video = [video]
conv_template = "qwen_1_5"
time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\nPlease describe this video in detail."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

cont = model.generate(
    input_ids,
    images=video,
    modalities= ["video"],
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print(text_outputs)

# Save output to output.json
with open("output.json", "w") as f:
    json.dump({"output": text_outputs}, f)