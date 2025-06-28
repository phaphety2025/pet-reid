
# Pet Re-Identification Gradio App

This Gradio app helps users compare two pet images or search for a lost pet by uploading a photo. It uses a Siamese Neural Network trained via fastai and PyTorch.

## Features

- 🔍 Compare two pet images  
- 📂 Upload one image to search a gallery for the most similar pets

## Usage

- Make sure to upload a trained `siamese_model.pkl` file  
- Ensure a folder named `gallery` exists with `.jpg` or `.png` images for similarity search  
- App is deployable on Hugging Face Spaces using `app.py`
