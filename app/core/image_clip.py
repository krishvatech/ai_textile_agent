import open_clip
from PIL import Image
import requests
import torch

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

def get_image_clip_embedding(image_url):
    img = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    img_input = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(img_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features[0].cpu().numpy().tolist()  # 512-dim

def get_text_clip_embedding(text):
    text_input = tokenizer([text])
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features[0].cpu().numpy().tolist()  # 512-dim
