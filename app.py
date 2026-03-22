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

# --- 3. Sidebar Configuration ---
st.sidebar.title("⚙️ AI Settings")
st.sidebar.write("Tweak the model's generation parameters:")
max_length = st.sidebar.slider("Max Caption Length", min_value=10, max_value=100, value=50, step=5)
creativity = st.sidebar.slider("AI Creativity (Temperature)", min_value=0.1, max_value=1.5, value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ using Streamlit, Hugging Face, and PyTorch.")

# --- 4. Main UI & Tabs ---
st.title("🚀 Advanced AI Image Captioning")

tab1, tab2, tab3 = st.tabs(["📸 Generate Captions", "🏗️ Architecture & Design", "💻 Run Locally"])

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
                # Prepare image
                inputs = processor(image, return_tensors="pt")
                # Generate text using sidebar parameters
                out = model.generate(
                    **inputs, 
                    max_new_tokens=max_length,
                    temperature=creativity,
                    do_sample=(creativity != 1.0) # Enable sampling if temperature is tweaked
                )
                caption = processor.decode(out[0], skip_special_tokens=True)
                
            st.success(f"**Generated Caption:** {caption.capitalize()}")

# ----- TAB 2: ARCHITECTURE -----
with tab2:
    st.header("🏗️ System Architecture & Design")
    st.write("""
    This application utilizes **Salesforce's BLIP (Bootstrapping Language-Image Pre-training)** architecture. 
    It is a unified vision-language model that achieves state-of-the-art results on a wide range of tasks.
    """)
    
    st.markdown("""
    ### 🧠 How the AI works
    1. **Vision Encoder (ViT):** The image is divided into patches (like a grid) and fed into a Vision Transformer. This extracts visual concepts (shapes, colors, objects). 
    2. **Text Decoder (BERT-based):** A language model takes those visual concepts and decodes them into a coherent English sentence, word by word, using Cross-Attention mechanisms.
    3. **Bootstrapping:** The model was trained on billions of noisy web images, filtering out the bad captions and learning from the high-quality ones.
    """)

# ----- TAB 3: RUN LOCALLY -----
with tab3:
    st.header("💻 How to Run This Project Locally")
    st.write("Want to run this lightning-fast on your own machine? Follow these steps in your terminal:")
    
    st.code("""
# 1. Clone the repository
git clone https://github.com/kulkarnisiddhesh2626/Advanced-Image-Captioning.git

# 2. Navigate into the project folder
cd Advanced-Image-Captioning

# 3. Install the required dependencies
pip install -r requirements.txt

# 4. Start the Streamlit server
streamlit run app.py
    """, language="bash")
    st.info("The app will automatically open in your web browser at http://localhost:8501")
