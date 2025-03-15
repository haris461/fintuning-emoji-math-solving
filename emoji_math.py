import streamlit as st
import torch
import os
import requests
import zipfile
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define model URL and directory
MODEL_URL = "https://github.com/haris461/fintuning-emoji-math-solving/releases/download/v1.0/model.zip"
MODEL_DIR = "./emoji-math-model"

# Function to download and extract model if not available
def download_and_extract_model():
    """Downloads and extracts the model if it doesn't exist locally."""
    zip_path = "./emoji-math-model.zip"

    if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

        st.write("Downloading model from GitHub Release...")
        response = requests.get(MODEL_URL, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

        st.write("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)

        os.remove(zip_path)  # Cleanup
        st.write("Model extraction complete.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    download_and_extract_model()  # Ensure model is available

    # 🔍 Debugging: Check model directory contents
    if not os.path.exists(MODEL_DIR):
        raise ValueError(f"❌ Model directory {MODEL_DIR} not found.")
    
    files = os.listdir(MODEL_DIR)
    if not files:
        raise ValueError(f"❌ Model directory is empty: {MODEL_DIR}")

    st.write(f"📂 Model directory contents: {files}")

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    except Exception as e:
        raise ValueError(f"❌ Model loading failed: {str(e)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Load the model
try:
    model, tokenizer, device = load_model()
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"🚨 Error loading model: {e}")
    st.stop()


