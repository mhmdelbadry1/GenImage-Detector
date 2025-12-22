#!/usr/bin/env python3
"""
Test script:
- Auto-detect all datasets except BigGAN
- Generate confusion matrix for each (saved as PNG)
- Special: nano_banana gets JSON only
"""
import torch
from torchvision import transforms
from PIL import Image
import os
import argparse
import json
from glob import glob
from src.models import get_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def test_on_dataset(model, model_name, test_dataset_path, device):
    """Test model on one dataset and return results"""
    dataset_name = os.path.basename(test_dataset_path)
    print(f"\nTesting on: {dataset_name}")
    
    # Find all images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(glob(os.path.join(test_dataset_path, '**', ext), recursive=True))
    
    if not image_files:
        print(f"  No images found, skipping")
        return None
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Predict
    all_preds = []
    all_labels = []  # All should be 1 (Fake)
    
    model.eval()
    with torch.no_grad():
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                output = model(img_tensor)
                _, pred = torch.max(output, 1)
                all_preds.append(pred.item())
                all_labels.append(1)  # True label is Fake
            except:
                continue
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = (all_preds == all_labels).sum() / len(all_labels) * 100
    
    print(f"  Samples: {len(all_labels)}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    return {
        "dataset": dataset_name,
        "samples": len(all_labels),
        "accuracy": float(accuracy),
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['resnet50', 'xception', 'vit'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_path = f"models/{args.model}/{args.model}_best.pth"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    print(f"Loading {args.model} from {model_path}")
    model = get_model(args.model, num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model = model.to(device)
    
    # Find all test datasets (exclude BigGAN)
    all_datasets = [d for d in glob('data/*') if os.path.isdir(d) and 'BigGAN' not in d]
    
    print(f"\nFound {len(all_datasets)} test datasets:")
    for ds in all_datasets:
        print(f"  - {os.path.basename(ds)}")
    
    # Test on each dataset
    all_results = {}
    
    for dataset_path in all_datasets:
        dataset_name = os.path.basename(dataset_path)
        result = test_on_dataset(model, args.model, dataset_path, device)
        
        if result is None:
            continue
            
        all_results[dataset_name] = result
        
        # Special handling for nano_banana: JSON only
        if 'nano_banana' in dataset_name.lower():
            json_path = f"models/{args.model}/nano_banana_results.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved JSON to {json_path}")
        else:
            # Generate confusion matrix
            cm = confusion_matrix(result['labels'], result['predictions'], labels=[0, 1])
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
            plt.title(f'{args.model.upper()} on {dataset_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            cm_path = f"models/{args.model}/confusion_{dataset_name}.png"
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved confusion matrix to {cm_path}")
    
    # Save summary
    summary_path = f"models/{args.model}/test_results_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
