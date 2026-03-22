# 🚀 Advanced AI Image Captioning

An end-to-end, production-ready Streamlit application that uses state-of-the-art Deep Learning to generate highly accurate captions for any image.

## 🌟 Features
* **State-of-the-Art Model**: Powered by Hugging Face and Salesforce's BLIP (Bootstrapping Language-Image Pre-training) Transformer architecture.
* **100+ Built-in Image Gallery**: Integrated with the Picsum API so users can test the AI dynamically without needing to download their own images.
* **Custom Parameter Tuning**: Users can tweak the AI's "creativity" (temperature) and caption length dynamically via the UI.
* **Optimized for CPU**: Engineered to run smoothly on free cloud tiers and local machines without requiring a GPU.

## 🏗️ Architecture
The system uses a unified vision-language model:
1. **Vision Transformer (ViT)**: Slices the image into patches to extract deep visual features.
2. **Cross-Attention Decoder**: Translates those visual embeddings into natural, coherent human language using a BERT-style architecture.

## 💻 How to Run Locally

If you want to run this application on your local machine for instant inference times, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/kulkarnisiddhesh2626/Advanced-Image-Captioning.git](https://github.com/kulkarnisiddhesh2626/Advanced-Image-Captioning.git)
