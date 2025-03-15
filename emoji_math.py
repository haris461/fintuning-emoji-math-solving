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

# Function to download and extract model if not available
def download_and_extract_model():
    zip_path = "emoji-math-model.zip"
    if not os.path.exists(MODEL_SUBDIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("./")
        os.remove(zip_path)
        print("Model extraction complete.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    download_and_extract_model()
    model = AutoModelForCausalLM.from_pretrained(MODEL_SUBDIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SUBDIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Load the model
model, tokenizer, device = load_model()

# Apply custom CSS styling
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
            background-color: #000000;
            color: white;
        }
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #FFD700;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #BBBBBB;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            border: 2px solid #FFD700;
            padding: 10px;
            color: white;
            font-size: 18px;
            background-color: #222222;
        }
        .stButton>button {
            background-color: #FFD700;
            color: black;
            font-size: 18px;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #FFC107;
        }
        [data-testid="stSidebar"] {
            background-color: #1E1E1E;
            color: white;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #BBBBBB;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.markdown("<h1 class='title'>✨ Emoji Math Solver ✨</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter an emoji-based math equation and get the solution.</p>", unsafe_allow_html=True)

# Sidebar with example equations
st.sidebar.header("🔹 Example Inputs")
st.sidebar.markdown("🚗 + 🚗 = 16")
st.sidebar.markdown("🐱 + 🐱 = 10")
st.sidebar.markdown("🍔 + 🍔 = 14")
st.sidebar.markdown("🏡 + 🏡 + 🏡 = 21")

# User input
equation = st.text_input("Enter the equation:", "🚗 + 🚗 + 🚗 + 🚗 = 20")

# Solve function
def solve_emoji_math(equation):
    model.eval()
    input_text = f"{equation} ->"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, num_beams=10, early_stopping=True, no_repeat_ngram_size=2, do_sample=False, temperature=0.1)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("->")[1].strip()

# Solve button
if st.button("Solve"):
    if "=" in equation and any(char.isdigit() for char in equation):
        solution = solve_emoji_math(equation)
        st.success(f"✅ Solution: {solution}")
    else:
        st.error("❌ Invalid equation format. Use something like '🚗 + 🚗 = 10'")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>💡 Developed with ❤️ using Streamlit</p>", unsafe_allow_html=True)

    
