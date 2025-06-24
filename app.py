
import gradio as gr
import torch
import torch.nn.functional as F
from fastai.vision.all import *
import glob

# Load model, define functions, etc. (your actual app logic goes here)
def greet(name):
    return "Hello " + name + "!!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
