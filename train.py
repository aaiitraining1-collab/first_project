import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from tqdm import tqdm

# Allowed target classes as requested by user
TARGET_CLASSES = ['glass', 'metal', 'paper', 'plastic']


class RecyclingDataset(Dataset):
    """
    Dataset loader for specific target classes in the recycling dataset folder structure.
    Recursively finds valid image files belonging to TARGET_CLASSES.
    """
    def __init__(self, data_root, target_classes=TARGET_CLASSES, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.target_classes = sorted(target_classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.target_classes)}
        
        self.samples = []
        
        # Locate directories for each class
        for cls_name in self.target_classes:
            # Check direct subfolder or dataset-resized subfolder
            cls_dirs = list(self.data_root.rglob(cls_name))
            valid_dirs = [d for d in cls_dirs if d.is_dir()]
            
            if not valid_dirs:
                print(f"[Warning] Class folder '{cls_name}' not found under {self.data_root}")
                continue
                
            cls_dir = valid_dirs[0]
            label_idx = self.class_to_idx[cls_name]
            
            # Find all images in class dir
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG'):
                for img_path in cls_dir.glob(ext):
                    self.samples.append((str(img_path), label_idx))
                    
        print(f"Found {len(self.samples)} images across {len(self.target_classes)} classes: {self.target_classes}")
        for cls_name in self.target_classes:
            idx = self.class_to_idx[cls_name]
            count = sum(1 for _, label in self.samples if label == idx)
            print(f"  - {cls_name}: {count} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Fallback black image
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label


def find_data_directory(base_dir="data"):
    """
    Auto-detect the dataset root folder under base_dir that contains the class subdirectories.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base data directory '{base_dir}' does not exist.")
        
    # Check if target classes are directly inside base_path
    if any((base_path / cls_name).exists() for cls_name in TARGET_CLASSES):
        return str(base_path)
        
    # Check subfolders (like dataset-resized)
    for sub in base_path.iterdir():
        if sub.is_dir() and any((sub / cls_name).exists() for cls_name in TARGET_CLASSES):
            return str(sub)
            
    return str(base_path)


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = find_data_directory(args.data_dir)
    print(f"Dataset root identified at: {data_dir}")

    # Image transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = RecyclingDataset(data_root=data_dir, target_classes=TARGET_CLASSES, transform=train_transform)
    
    if len(full_dataset) == 0:
        print("Error: No images found in dataset directory!")
        sys.exit(1)

    # Train / Validation Split (80% / 20%)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Override val dataset transform
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Dataset Split: {train_size} training samples, {val_size} validation samples.")

    # Load Pre-trained Model (MobileNetV3 Small for speed and accuracy)
    print("Initializing MobileNetV3 pre-trained model...")
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        # Fallback if weights argument syntax varies across torchvision versions
        model = models.mobilenet_v3_small(pretrained=True)

    # Modify final classifier layer for 4 classes
    num_classes = len(TARGET_CLASSES)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    print(f"\n--- Starting Training ({args.epochs} Epochs) ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]  "):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        print(f"Epoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}%")
        print(f"  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc * 100:.2f}%")

        # Save Best Model Checkpoint
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'class_names': TARGET_CLASSES,
                'class_to_idx': full_dataset.class_to_idx
            }, args.output_model)
            print(f"  --> Saved new best model checkpoint to {args.output_model} (Val Acc: {epoch_val_acc * 100:.2f}%)")

    # Save Class Names JSON
    with open(args.output_labels, 'w') as f:
        json.dump({
            'class_names': TARGET_CLASSES,
            'class_to_idx': full_dataset.class_to_idx
        }, f, indent=2)
    print(f"\nSaved class mappings to {args.output_labels}")
    print(f"Training Complete! Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Trained model saved at: {os.path.abspath(args.output_model)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an Image Classifier on Recycling Data (glass, metal, paper, plastic)")
    parser.add_argument('--data_dir', type=str, default='data', help="Path to data directory")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--output_model', type=str, default='recycling_model.pth', help="File path to save trained model")
    parser.add_argument('--output_labels', type=str, default='class_names.json', help="File path to save class labels")

    args = parser.parse_args()
    train_model(args)
