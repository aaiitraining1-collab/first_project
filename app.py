import os
import json
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

MODEL_PATH = 'recycling_model.pth'
LABELS_PATH = 'class_names.json'

app = FastAPI(
    title="Smart Recycling Companion API",
    description="REST API & Web Interface for image-based recycling classification",
    version="1.0.0"
)

# Mount static directory and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global model and labels
class_names = []
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_trained_model():
    global class_names, model
    
    if not os.path.exists(LABELS_PATH):
        print(f"Warning: Label mapping file '{LABELS_PATH}' not found. Using defaults.")
        class_names = ["glass", "metal", "paper", "plastic"]
    else:
        with open(LABELS_PATH, 'r') as f:
            meta = json.load(f)
        class_names = meta['class_names']
    
    # Architecture definition
    net = models.mobilenet_v3_small()
    in_features = net.classifier[3].in_features
    net.classifier[3] = nn.Linear(in_features, len(class_names))
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            net.load_state_dict(checkpoint)
        print(f"Loaded model weights from {MODEL_PATH}")
    else:
        print(f"Warning: '{MODEL_PATH}' not found. Initializing unweighted model.")
        
    net = net.to(device)
    net.eval()
    model = net

# Initialize model at module load
load_trained_model()

# Image Preprocessing Transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """Serves the main web user interface."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "categories": class_names
    }

@app.get("/api/sample-image/{sample_type}")
async def get_sample_image(sample_type: str):
    """Serves a sample dataset image for testing."""
    sample_type = sample_type.lower()
    
    sample_paths = {
        "glass": "data/dataset-resized/glass/glass1.jpg",
        "metal": "data/dataset-resized/metal/metal1.jpg",
        "paper": "data/dataset-resized/paper/paper1.jpg",
        "plastic": "data/dataset-resized/plastic/plastic1.jpg"
    }
    
    if sample_type in sample_paths and os.path.exists(sample_paths[sample_type]):
        return FileResponse(sample_paths[sample_type], media_type="image/jpeg")
    
    # Return placeholder synthetic image if file is missing
    img = Image.new('RGB', (224, 224), color=(16, 185, 129))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return HTMLResponse(content=buf.getvalue(), media_type="image/jpeg")

@app.post("/api/predict")
async def predict_recycling_category(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, processes it through PyTorch MobileNetV3,
    and returns sorted predictions.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Preprocess image
    tensor = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Format predictions sorted by probability descending
    top_probs, top_indices = torch.topk(probabilities, k=len(class_names))
    
    predictions = []
    for idx, prob in zip(top_indices, top_probs):
        cls_name = class_names[idx.item()]
        conf_val = float(prob.item())
        predictions.append({
            "class_name": cls_name.capitalize(),
            "confidence_pct": round(conf_val * 100, 2),
            "score": round(conf_val, 4)
        })

    top_conf_pct = predictions[0]["confidence_pct"]
    is_recyclable = top_conf_pct >= 70.0
    display_result = predictions[0]["class_name"] if is_recyclable else "Non-recyclable"

    return {
        "filename": file.filename,
        "is_recyclable": is_recyclable,
        "result_status": display_result,
        "top_prediction": predictions[0],
        "predictions": predictions
    }

if __name__ == '__main__':
    import uvicorn
    print("Starting EcoScan AI Web Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
