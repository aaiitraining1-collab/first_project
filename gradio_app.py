import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

MODEL_PATH = 'recycling_model.pth'
LABELS_PATH = 'class_names.json'

CATEGORY_META = {
    "glass": {
        "icon": "🍾",
        "tips": [
            "Rinse out food or beverage residues before recycling.",
            "Separate metal and plastic caps/lids.",
            "Avoid mixing cookware (Pyrex) or window glass with container glass."
        ]
    },
    "metal": {
        "icon": "🥫",
        "tips": [
            "Rinse food tins and aluminum drink cans.",
            "Crush cans if possible to save bin space.",
            "Clean aluminum foil can be rolled into a ball and recycled."
        ]
    },
    "paper": {
        "icon": "📦",
        "tips": [
            "Flatten cardboard boxes to save space.",
            "Keep paper clean and dry (greasy pizza boxes go in trash).",
            "Remove plastic sleeves, bubble wrap, and non-paper tape."
        ]
    },
    "plastic": {
        "icon": "🥤",
        "tips": [
            "Check recycling numbers (#1 PET and #2 HDPE are widely accepted).",
            "Rinse containers thoroughly.",
            "Reattach caps securely to empty plastic bottles."
        ]
    }
}

def load_model_and_labels():
    if not os.path.exists(LABELS_PATH):
        class_names = ["glass", "metal", "paper", "plastic"]
    else:
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
    return model, class_names, device

model, class_names, device = load_model_and_labels()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify_recycling_image(img):
    if img is None:
        return "Please upload or select an image to classify."
        
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img).convert('RGB')
    else:
        img = img.convert('RGB')

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top_prob, top_idx = torch.topk(probabilities, k=len(class_names))

    best_class = class_names[top_idx[0].item()]
    best_conf = top_prob[0].item() * 100
    
    is_recyclable = best_conf >= 70.0
    cat_key = best_class.lower()
    meta = CATEGORY_META.get(cat_key, {"icon": "♻️", "tips": []})
    
    if is_recyclable:
        tips_md = "\n".join([f"- {tip}" for tip in meta["tips"]])
        summary_md = f"""
## ♻️ Result: **{meta['icon']} {best_class.capitalize()}**
- **Detected Material:** {best_class.capitalize()}

---
### 💡 Disposal & Recycling Tips for {best_class.capitalize()}:
{tips_md}
"""
    else:
        summary_md = f"""
## 🚫 Result: **Non-recyclable**
- **Status:** Non-recyclable

---
### 💡 General Disposal Guidelines:
- The item cannot be confidently identified as recyclable glass, metal, paper, or plastic.
- Please dispose of this item in the **general waste / trash bin**.
"""
    return summary_md

# Sample dataset image paths for quick testing in Gradio UI
example_images = [
    ["data/dataset-resized/glass/glass1.jpg"],
    ["data/dataset-resized/metal/metal1.jpg"],
    ["data/dataset-resized/paper/paper1.jpg"],
    ["data/dataset-resized/plastic/plastic1.jpg"]
]
example_images = [ex for ex in example_images if os.path.exists(ex[0])]

with gr.Blocks(title="Smart Recycling Companion") as demo:
    gr.Markdown("""
    # ♻️ Smart Recycling Companion
    Upload an image of a waste item to classify whether it is **Plastic**, **Metal**, **Glass**, **Paper**, or **Non-recyclable**.
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload or Select Image")
            submit_btn = gr.Button("Classify Image 🔍", variant="primary")
            
            if example_images:
                gr.Examples(
                    examples=example_images,
                    inputs=input_image,
                    label="Sample Test Images"
                )
                
        with gr.Column():
            summary_output = gr.Markdown(label="Classification Result & Guidance")
            
    submit_btn.click(
        fn=classify_recycling_image,
        inputs=input_image,
        outputs=summary_output
    )
    input_image.change(
        fn=classify_recycling_image,
        inputs=input_image,
        outputs=summary_output
    )

if __name__ == '__main__':
    print("Launching Gradio app on http://127.0.0.1:7860 ...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
