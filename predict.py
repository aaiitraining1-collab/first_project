import argparse
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def predict_image(image_path, model_path='recycling_model.pth', labels_path='class_names.json'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label mapping
    with open(labels_path, 'r') as f:
        meta = json.load(f)
    class_names = meta['class_names']
    
    # Load model state
    checkpoint = torch.load(model_path, map_location=device)
    
    model = models.mobilenet_v3_small()
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(class_names))
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()

    # Define validation image transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_prob, top_idx = torch.topk(probabilities, k=len(class_names))
    best_class = class_names[top_idx[0].item()]
    best_confidence = top_prob[0].item() * 100
    
    result = best_class.upper() if best_confidence >= 70.0 else "NON-RECYCLABLE"
    
    print(f"\nPrediction Result for image: '{image_path}'")
    print("-" * 45)
    print(f"Result: {result}\n")
        
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify a recycling image (glass, metal, paper, plastic)")
    parser.add_argument('--image', type=str, required=True, help="Path to input image")
    parser.add_argument('--model', type=str, default='recycling_model.pth', help="Path to trained .pth model file")
    parser.add_argument('--labels', type=str, default='class_names.json', help="Path to class labels json file")

    args = parser.parse_args()
    predict_image(args.image, args.model, args.labels)
