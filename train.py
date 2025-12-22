#!/usr/bin/env python3
"""
Clean training script:
- Train ONLY on BigGAN
- Save model to models/<model_name>/
- Generate PCA before and after training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
from src.dataset import GenImageDataset
from src.models import get_model
from src.analyze_pca import run_pca_analysis
from sklearn.metrics import accuracy_score, f1_score

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['resnet50', 'xception', 'vit'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training {args.model} on BigGAN ONLY")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 1. PCA BEFORE training (pre-trained features)
    print("Step 1: PCA Analysis (BEFORE training)")
    os.makedirs(f"models/{args.model}", exist_ok=True)
    run_pca_analysis('data/BigGAN', 
                     output_file=f'models/{args.model}/pca_before_training.png')
    
    # 2. Load BigGAN data
    print("\nStep 2: Loading BigGAN dataset")
    train_ds = GenImageDataset('data/BigGAN', split='train')
    val_ds = GenImageDataset('data/BigGAN', split='val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    
    # 3. Create model
    print(f"\nStep 3: Initializing {args.model}")
    model = get_model(args.model, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 4. Training loop
    print(f"\nStep 4: Training for {args.epochs} epochs")
    best_val_acc = 0.0
    best_model_wts = None
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}/{args.epochs-1} - "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    
    # 5. Save model
    print(f"\nStep 5: Saving model (Best Val Acc: {best_val_acc:.4f})")
    model_path = f"models/{args.model}/{args.model}_best.pth"
    torch.save(best_model_wts, model_path)
    print(f"Saved to {model_path}")
    
    # 6. Save metadata
    metadata = {
        "model_name": args.model,
        "train_dataset": "BigGAN",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "epochs": args.epochs,
        "best_val_accuracy": float(best_val_acc)
    }
    with open(f"models/{args.model}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 7. PCA AFTER training (trained features)
    print("\nStep 6: PCA Analysis (AFTER training)")
    # TODO: Extract features using trained model and plot PCA
    print("(Will implement PCA with trained features)")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
