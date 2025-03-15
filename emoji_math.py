import streamlit as st
import torch
import os
import requests
import zipfile
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast

# Ensure proper asyncio event loop handling
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

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

# Apply custom CSS styling
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
            font-family: Arial, sans-serif;
        }
        .stApp {
            max-width: 800px;
            margin: auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stTextInput, .stButton {
            width: 100%;
        }
        .stTextArea {
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("🔢 Emoji Math Solver")
st.write("Enter an emoji-based math problem, and let AI solve it!")

user_input = st.text_input("Enter Emoji Math Expression:")

if st.button("Solve"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_length=50)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        st.success(f"Solution: {result}")
    else:
        st.warning("Please enter a valid emoji math expression.")
