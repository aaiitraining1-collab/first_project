import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(
    page_title="Smart Recycling Companion",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = 'recycling_model.pth'
LABELS_PATH = 'class_names.json'

# Material metadata and recycling guidance
CATEGORY_META = {
    "glass": {
        "icon": "🍾",
        "title": "Glass",
        "recyclable": True,
        "tips": [
            "Rinse out food or beverage residues before recycling.",
            "Separate metal and plastic caps/lids to recycle independently.",
            "Avoid mixing cookware (Pyrex) or window glass with container glass."
        ]
    },
    "metal": {
        "icon": "🥫",
        "title": "Metal",
        "recyclable": True,
        "tips": [
            "Rinse food tins and aluminum drink cans.",
            "Crush cans if possible to save bin space.",
            "Clean aluminum foil can be rolled into a ball and recycled."
        ]
    },
    "paper": {
        "icon": "📦",
        "title": "Paper",
        "recyclable": True,
        "tips": [
            "Flatten cardboard boxes to save space.",
            "Keep paper clean and dry (greasy pizza boxes go in trash).",
            "Remove plastic sleeves, bubble wrap, and non-paper tape."
        ]
    },
    "plastic": {
        "icon": "🥤",
        "title": "Plastic",
        "recyclable": True,
        "tips": [
            "Check recycling numbers (#1 PET and #2 HDPE are widely accepted).",
            "Rinse containers thoroughly.",
            "Reattach caps securely to empty plastic bottles."
        ]
    }
}

@st.cache_resource
def load_model_and_labels():
    if not os.path.exists(LABELS_PATH):
        st.error(f"Label mapping file '{LABELS_PATH}' not found.")
        return None, ["glass", "metal", "paper", "plastic"]
        
    with open(LABELS_PATH, 'r') as f:
        meta = json.load(f)
    class_names = meta['class_names']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.mobilenet_v3_small()
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(class_names))
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
    model = model.to(device)
    model.eval()
    return model, class_names

model, class_names = load_model_and_labels()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">♻️ Smart Recycling Companion</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an image to classify whether it is <b>Plastic</b>, <b>Metal</b>, <b>Glass</b>, or <b>Paper</b> with AI confidence percentages.</p>', unsafe_allow_html=True)

# Sidebar with model info & instructions
with st.sidebar:
    st.header("ℹ️ App Information")
    st.markdown("""
    **Model Architecture:** PyTorch MobileNetV3 Small  
    **Supported Categories:**
    - 🍾 **Glass**
    - 🥫 **Metal**
    - 📦 **Paper**
    - 🥤 **Plastic**
    """)
    st.divider()
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Upload an image (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) or select a sample image button.
    2. View whether the item is classified as **Plastic**, **Metal**, **Glass**, **Paper**, or **Non-recyclable**.
    """)

# Main page layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Input Image")
    
    st.write("**Quick Test Samples:**")
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
    
    selected_sample = None
    if sample_col1.button("🍾 Glass", use_container_width=True):
        selected_sample = "data/dataset-resized/glass/glass1.jpg"
    if sample_col2.button("🥫 Metal", use_container_width=True):
        selected_sample = "data/dataset-resized/metal/metal1.jpg"
    if sample_col3.button("📦 Paper", use_container_width=True):
        selected_sample = "data/dataset-resized/paper/paper1.jpg"
    if sample_col4.button("🥤 Plastic", use_container_width=True):
        selected_sample = "data/dataset-resized/plastic/plastic1.jpg"
        
    uploaded_file = st.file_uploader("Or choose an image file from your computer...", type=["jpg", "jpeg", "png", "bmp", "webp"])
    
    image_to_process = None
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file).convert('RGB')
    elif selected_sample and os.path.exists(selected_sample):
        image_to_process = Image.open(selected_sample).convert('RGB')
        
    if image_to_process:
        st.image(image_to_process, caption="Uploaded Input Image", use_container_width=True)

with col2:
    st.subheader("2. Classification Results")
    
    if image_to_process is None:
        st.info("👈 Upload an image or click a sample image button on the left to run classification.")
    elif model is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor = transform(image_to_process).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        top_prob, top_idx = torch.topk(probabilities, k=len(class_names))
        
        top_class = class_names[top_idx[0].item()]
        top_confidence = top_prob[0].item() * 100
        cat_key = top_class.lower()
        meta = CATEGORY_META.get(cat_key, {"icon": "♻️", "title": top_class.capitalize(), "tips": []})
        
        is_recyclable = top_confidence >= 70.0
        
        if is_recyclable:
            st.success(f"### Result: **{meta['icon']} {top_class.upper()}**")
            st.metric(label="Detected Material", value=f"{meta['icon']} {top_class.capitalize()}")
                
            st.divider()
            st.markdown(f"### 💡 Recycling Tips for {top_class.capitalize()}")
            for tip in meta['tips']:
                st.markdown(f"- {tip}")
        else:
            st.error("### Result: **🚫 NON-RECYCLABLE**")
            st.metric(label="Result Status", value="🚫 Non-recyclable")
                
            st.divider()
            st.markdown("### 💡 General Disposal Guidelines")
            st.markdown("- This item cannot be confidently identified as recyclable glass, metal, paper, or plastic.")
            st.markdown("- Please dispose of this item in the **general waste / trash bin**.")
            st.markdown("- Check local waste guidelines for unclassified items.")
