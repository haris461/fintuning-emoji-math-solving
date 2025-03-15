import streamlit as st
import torch
import os
import requests
import zipfile
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast

# Define model URL and directory
MODEL_URL = "https://github.com/haris461/fintuning-emoji-math-solving/releases/download/v1.0/emoji-math-model.zip"
MODEL_DIR = "./emoji-math-model"
MODEL_SUBDIR = os.path.join(MODEL_DIR, "checkpoint-108")  # Adjusted path

print("Model path:", os.path.abspath(MODEL_SUBDIR))

# Debugging: Check model directory files
if os.path.exists(MODEL_DIR):
    print("Files in model directory:", os.listdir(MODEL_DIR))
if os.path.exists(MODEL_SUBDIR):
    print("Files in checkpoint directory:", os.listdir(MODEL_SUBDIR))
else:
    print("Checkpoint folder missing")

# Ensure an event loop is running
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

# Function to download and extract model if not available
def download_and_extract_model():
    """Downloads and extracts the model if it doesn't exist locally."""
    zip_path = "emoji-math-model.zip"
    
    if not os.path.exists(MODEL_SUBDIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        print("Downloading model from GitHub Release...")
        response = requests.get(MODEL_URL, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

        print("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("./")
        
        os.remove(zip_path)  # Cleanup
        print("Model extraction complete.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    download_and_extract_model()  # Ensure model is available
    
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_SUBDIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_SUBDIR)
    except OSError:
        print("Error loading tokenizer from checkpoint. Falling back to manually loading tokenizer.")
        tokenizer = GPT2TokenizerFast(
            vocab_file="./emoji-math-model/vocab.json",
            merges_file="./emoji-math-model/merges.txt"
        )
        model = AutoModelForCausalLM.from_pretrained(MODEL_SUBDIR)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Load the model
model, tokenizer, device = load_model()
