import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI Image Captioner", page_icon="🖼️", layout="centered")

# --- 2. Load the Hugging Face Model (Cached for Speed) ---
@st.cache_resource
def load_hf_model():
    # We use Salesforce's BLIP model - fast, lightweight, and highly accurate
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_hf_model()

# --- 3. Streamlit UI & Text ---
st.title("🖼️ AI Image Captioning")
st.write("Powered by Hugging Face & Salesforce BLIP")

with st.expander("ℹ️ About this App", expanded=False):
    st.write("""
        This application uses a state-of-the-art **Transformer** model called BLIP (Bootstrapping Language-Image Pre-training) built by Salesforce. 
        Unlike traditional CNN-RNN architectures, this model was pre-trained on millions of image-text pairs, allowing it to understand complex scenes and generate highly accurate human-like descriptions in milliseconds.
    """)

# --- 4. Image Uploader ---
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to RGB to prevent errors with transparent PNGs
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display the image nicely
    st.image(image, caption="Your Uploaded Image", use_column_width=True)
    
    # Generate the caption
    with st.spinner("Generating caption using Hugging Face..."):
        # Prepare the image for the model
        inputs = processor(image, return_tensors="pt")
        
        # Generate the text tokens
        out = model.generate(**inputs, max_new_tokens=50)
        
        # Decode the tokens back into human text
        caption = processor.decode(out[0], skip_special_tokens=True)
        
    st.success(f"**Generated Caption:** {caption.capitalize()}")
