import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
from io import BytesIO

# --- 1. Page Configuration ---
st.set_page_config(page_title="Advanced AI Captioner", page_icon="🚀", layout="wide")

# --- 2. Load the AI Brain (Cached) ---
@st.cache_resource
def load_hf_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_hf_model()

# --- 3. Sidebar Configuration & Explanations ---
st.sidebar.title("⚙️ AI Settings")
st.sidebar.write("Tweak the model's generation parameters:")

# Parameter 1: Max Length
max_length = st.sidebar.slider("Max Caption Length", min_value=10, max_value=100, value=50, step=5)
st.sidebar.caption("**Max Length:** Controls the absolute limit of words the AI is allowed to generate. A higher number allows for longer, more descriptive sentences.")

# Parameter 2: Creativity (Temperature)
creativity = st.sidebar.slider("AI Creativity (Temperature)", min_value=0.1, max_value=1.5, value=1.0, step=0.1)
st.sidebar.caption("**Creativity (Temperature):** A low value (e.g., 0.1) makes the AI strict, factual, and predictable. A high value (e.g., 1.5) allows the AI to take risks, use varied vocabulary, and be more 'creative' with its descriptions.")

st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ using Streamlit, Hugging Face, and PyTorch.")

# --- 4. Main UI & Tabs ---
st.title("🚀 Advanced AI Image Captioning")

# Reduced to two clean, professional tabs
tab1, tab2 = st.tabs(["📸 Generate Captions", "🏗️ Architecture & Design"])

# ----- TAB 1: THE APP -----
with tab1:
    st.write("Upload your own image, or choose from our built-in gallery of 100+ images!")
    
    input_method = st.radio("Choose Input Method:", ("Upload an Image", "Select from Built-in Gallery"))
    
    image = None
    
    if input_method == "Upload an Image":
        uploaded_file = st.file_uploader("Drag and drop your image here...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            
    elif input_method == "Select from Built-in Gallery":
        image_id = st.slider("Select an image ID (1 to 100):", 1, 100, 10)
        # Fetch high-quality random images via an API to save repo space!
        img_url = f"https://picsum.photos/id/{image_id}/800/600"
        try:
            response = requests.get(img_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except:
            st.error("Failed to load image from server. Please try another ID.")
            
    if image is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Target Image", use_column_width=True)
            
        with col2:
            st.subheader("AI Analysis")
            with st.spinner("Decoding image features..."):
                inputs = processor(image, return_tensors="pt")
                
                # Apply the sidebar settings to the AI's generation function
                out = model.generate(
                    **inputs, 
                    max_new_tokens=max_length,
                    temperature=creativity,
                    do_sample=(creativity != 1.0) # Required by Hugging Face to use temperature
                )
                caption = processor.decode(out[0], skip_special_tokens=True)
                
            st.success(f"**Generated Caption:** {caption.capitalize()}")

# ----- TAB 2: ARCHITECTURE -----
with tab2:
    st.header("🏗️ System Architecture & Design")
    st.write("""
    This application utilizes **Salesforce's BLIP (Bootstrapping Language-Image Pre-training)** architecture. 
    It is a unified vision-language model that achieves state-of-the-art results by merging visual feature extraction with natural language processing.
    """)
    
    
    
    st.markdown("""
    ### 🧠 How the AI Decodes an Image
    1. **Vision Encoder (ViT):** The image is first divided into a grid of distinct patches. These patches are fed into a Vision Transformer (ViT) which learns to recognize spatial relationships, colors, and objects.
    2. **Cross-Attention Mechanism:** The AI does not just look at the image; it maps the visual features directly to text tokens. It pays "attention" to specific parts of the image when predicting specific words.
    3. **Text Decoder:** A BERT-based language model takes those visual embeddings and generates a coherent English sentence word-by-word based on the temperature and length constraints set by the user.
    """)
