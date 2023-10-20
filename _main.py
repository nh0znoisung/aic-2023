import torch
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


print(get_score("../docs/_static/merlion.png", "that is dog"))