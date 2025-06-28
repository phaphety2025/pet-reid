import gradio as gr
import torch
from fastai.learner import load_learner
import torch.nn.functional as F
from fastai.vision.all import *
import glob
import os

class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        body = create_body(resnet18, pretrained=False)
        self.encoder = nn.Sequential(AdaptiveConcatPool2d(), Flatten(), nn.Linear(512, 256))
        self.backbone = body

    def forward(self, x1, x2):
        f1 = self.encoder(self.backbone(x1))
        f2 = self.encoder(self.backbone(x2))
        return F.pairwise_distance(f1, f2)

base_path = "gallery"

def preprocess(img_path):
    img = PILImage.create(img_path)
    return img.to_tensor().unsqueeze(0)

def predict_pair_gr(img1_path, img2_path):
    with torch.no_grad():
        d = model(preprocess(img1_path), preprocess(img2_path)).item()
    return f"Similarity Distance: {d:.4f} (Lower = More Similar)"

def search_similar_gr(query_img_path, top_k=3):
    gallery_paths = glob.glob(f"{base_path}/*.jpg") + glob.glob(f"{base_path}/*.png")
    if not gallery_paths:
        return ["No gallery images found."]
    query = preprocess(query_img_path)
    with torch.no_grad():
        q_embed = model.encoder(query)
        results = [(F.pairwise_distance(q_embed, model.encoder(preprocess(p))).item(), p) for p in gallery_paths]
    results.sort()
    return [p for _, p in results[:top_k]]

with gr.Blocks() as demo:
    with gr.Tab("🔍 Compare Two Pets"):
        gr.Interface(
            fn=predict_pair_gr,
            inputs=[
                gr.Image(type="filepath", label="Pet Image 1"),
                gr.Image(type="filepath", label="Pet Image 2")
            ],
            outputs=gr.Textbox(label="Similarity Score")
        ).render()

    with gr.Tab("Search Lost Pet"):
        gr.Interface(
            fn=search_similar_gr,
            inputs=gr.Image(type="filepath", label="Upload Lost Pet Image"),
            outputs=gr.Gallery(label="Top Matches")
        ).render()

demo.launch(share=True, debug=False)