
import gradio as gr
import torch
import torch.nn.functional as F
from fastai.vision.all import *
import glob

# Load your model and preprocessing here
learn = load_learner("siamese_model.pth")
model = learn.model

base_path = "."  # Adjust if needed

def preprocess(img_path):
    return PILImage.create(img_path).to_tensor().unsqueeze(0)

def predict_pair_gr(img1_path, img2_path):
    with torch.no_grad():
        d = model(preprocess(img1_path), preprocess(img2_path)).item()
    return f"Similarity Distance: {d:.4f} (Lower = More Similar)"

def search_similar_gr(query_img_path, top_k=3):
    gallery = glob.glob(f"{base_path}/gallery/*.jpg")
    query = preprocess(query_img_path)

    with torch.no_grad():
        q_embed = model.encoder(query)
        results = [
            (F.pairwise_distance(q_embed, model.encoder(preprocess(p))).item(), p)
            for p in gallery
        ]
    return [p for _, p in sorted(results)[:top_k]]

with gr.Blocks() as demo:
    with gr.Tab("🔍 Compare Two Pets"):
        gr.Interface(
            fn=predict_pair_gr,
            inputs=[gr.Image(type="filepath", label="Pet Image 1"), gr.Image(type="filepath", label="Pet Image 2")],
            outputs=gr.Textbox(label="Similarity Score"),
        ).render()

    with gr.Tab("📂 Search Lost Pet"):
        gr.Interface(
            fn=search_similar_gr,
            inputs=gr.Image(type="filepath", label="Upload a Photo of Lost Pet"),
            outputs=gr.Gallery(label="Top Matches")
        ).render()

demo.launch()
