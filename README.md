# 🚀 Advanced AI Image Captioning

An end-to-end, production-ready Streamlit application that uses state-of-the-art Deep Learning to generate highly accurate captions for any image.

## 🌟 Application Features
* **State-of-the-Art Model**: Powered by Hugging Face and Salesforce's BLIP (Bootstrapping Language-Image Pre-training) Transformer architecture.
* **100+ Built-in Image Gallery**: Integrated with the Picsum API. Users can test the AI dynamically using a slider without needing to download or source their own images.
* **Dynamic AI Settings**: 
  * **Temperature Control**: Adjust the AI's creativity level. Lower values produce strict, factual captions, while higher values encourage creative and varied vocabulary.
  * **Length Limits**: Dynamically control the absolute maximum word count the AI is allowed to output.

## 🏗️ Technical Architecture
The system uses a unified vision-language approach:
1. **Vision Transformer (ViT)**: Slices the image into patches to extract deep visual features and spatial relationships.
2. **Cross-Attention Decoder**: Translates visual embeddings into natural, coherent human language using a BERT-style text generation process.

## 🛠️ Tech Stack
* **Frontend UI**: Streamlit
* **Deep Learning Framework**: PyTorch
* **Model Hub**: Hugging Face `transformers`
* **Image Processing**: Pillow (PIL) & Requests
