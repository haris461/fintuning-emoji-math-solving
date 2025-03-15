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

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        print("Downloading model from GitHub Release...")
        response = requests.get(MODEL_URL, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

        print("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        
        os.remove(zip_path)  # Cleanup
        print("Model extraction complete.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    download_and_extract_model()  # Ensure model is available
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Load the model
model, tokenizer, device = load_model()

# Apply dark mode CSS styles with gold accents
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

        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {
            color: white;
        }

        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #FFD700;
        }

        .white-number {
            color: #FFFFFF !important;
        }

        .stAlert {
            background-color: #222222 !important;
            color: #FFD700 !important;
            border: 2px solid #FFD700;
        }

        .footer {
            text-align: center;
            font-size: 14px;
            color: #BBBBBB;
            margin-top: 20px;
        }

        .stText, .stMarkdown, .stTextInput label {
            color: white !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)

# Inference function
def solve_emoji_math(equation):
    model.eval()
    input_text = f"{equation} ->"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=10,
            early_stopping=True,
            no_repeat_ngram_size=2,
            do_sample=False,
            temperature=0.1
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    solution = result.split("->")[1].strip()
    
    # Post-processing for mathematical accuracy
    emoji = equation.split()[0]
    count = equation.count(emoji)
    total = int(equation.split("=")[1].strip())
    expected_value = total // count
    if f"{emoji} = {expected_value}" != solution:
        solution = f"{emoji} = {expected_value}"
    return solution

# Streamlit UI
st.markdown("<h1 class='title'>✨ Emoji Math Solver ✨</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter an emoji-based math equation and get the solution.</p>", unsafe_allow_html=True)

# User input
equation = st.text_input("Enter the equation:", "🚗 + 🚗 + 🚗 + 🚗 = 20")

if st.button("Solve"):
    if "=" in equation and any(char.isdigit() for char in equation):
        solution = solve_emoji_math(equation)
        st.success(f"✅ Solution: {solution}")
    else:
        st.error("❌ Invalid equation format. Use something like '🚗 + 🚗 = 10'")

# Example Equations in Sidebar with white numbers
st.sidebar.header("🔹 Example Inputs")
st.sidebar.markdown("🚗 + 🚗 = <span class='white-number'>16</span>", unsafe_allow_html=True)
st.sidebar.markdown("🐱 + 🐱 = <span class='white-number'>10</span>", unsafe_allow_html=True)
st.sidebar.markdown("🍔 + 🍔 = <span class='white-number'>14</span>", unsafe_allow_html=True)
st.sidebar.markdown("🏡 + 🏡 + 🏡 = <span class='white-number'>21</span>", unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>💡 Developed with ❤️ using Streamlit</p>", unsafe_allow_html=True)

