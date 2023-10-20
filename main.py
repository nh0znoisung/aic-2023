import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
import argparse
import glob
import os
from itertools import chain
import pandas as pd
import csv
# ! pip install salesforce-lavis
# sudo apt-get install unzip
# 400.962482213974s - 7,658 - 202,148

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)
model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)
# model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "base", device=device, is_eval=True)
# model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
# model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)

def get_score(path: str, caption: str):
    # "../docs/_static/merlion.png"
    # caption = "that is dog"
    raw_image = Image.open(path).convert("RGB")
    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    txt = text_processors["eval"](caption)

    itm_output = model({"image": img, "text_input": txt}, match_head="itm")
    itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
    score = itm_scores[:, 1].item()
    
    return score



parser = argparse.ArgumentParser(description="Read caption from file")
parser.add_argument("--caption", help="Path argument of caption")
args = parser.parse_args()

caption_path = args.caption # queries/query-x.txt
caption_file = caption_path.split("/")[-1]
caption_name = caption_file.split(".")[0]
print('Caption path: ', caption_path)
print('Caption file: ', caption_file)
print('Caption name: ', caption_name)

caption = None
with open(caption_path, "r") as file:
    caption = file.read()
print('Caption: ', caption)

# Load DF
df = pd.read_csv('merged_keyframes.csv')
print(df.head(5))

# Process
df['score'] = df['path'].apply(lambda path: get_score(path, caption))

# Sort
result = df.sort_values(by='score', ascending=False).head(100)
print(result.head(10))

result.to_csv(f'submission-lake/{caption_name}.csv', index=False)
result[['video_name', 'frame_idx']].to_csv(f'submission/{caption_name}.csv', index=False)
print('Save sucessfully!!!')


# RuntimeError: CUDA error: out of memory
# CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
# Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

# 5tab là max