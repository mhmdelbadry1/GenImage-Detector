import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import GenImageDataset
from src.models import get_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def run_pca_analysis(data_dir, output_file='plots/pca_visualization.png', samples=500):
    print("--- Starting PCA Analysis ---")
    
    # 1. Setup Dataset
    # We want a random subset, so we scan the whole folder and just take the first N or shuffle
    # GenImageDataset finds everything recursively.
    dataset = GenImageDataset(data_dir, split=None)
    
    if len(dataset) == 0:
        print("Error: No images found.")
        return
        
    # Limit samples for speed if needed, but PCA on 1000 is fast.
    if len(dataset) > samples:
        indices = torch.randperm(len(dataset))[:samples]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 2. Setup Feature Extractor (ResNet50)
    # Force CPU to avoid OOM since another process is using GPU
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model and strip head
    model = get_model('resnet50', pretrained=True).to(device)
    # timm models usually have a 'dataset' layer or 'fc' layer. 
    # For feature extraction in timm, we can use forward_features() but it returns specific maps.
    # Simpler: replace the fully connected layer with Identity
    model.reset_classifier(0) # Removes the final layer (global pool remains? No, usually pool is distinct)
    # Actually reset_classifier(0) in timm leaves the pooling, so output is (B, NumFeatures). perfect.
    
    model.eval()
    
    # 3. Extract Features
    features_list = []
    labels_list = []
    
    print("Extracting features...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # Forward pass
            feats = model(images) 
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
            
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    print(f"Extracted features shape: {features.shape}")
    
    # 4. Compute PCA
    print("Computing PCA...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    # 5. Plot
    print(f"Plotting to {output_file}...")
    plt.figure(figsize=(10, 8))
    
    # Class 0: Real (Nature)
    idx_0 = labels == 0
    plt.scatter(features_pca[idx_0, 0], features_pca[idx_0, 1], c='blue', label='Real (Nature)', alpha=0.6)
    
    # Class 1: Fake (AI)
    idx_1 = labels == 1
    plt.scatter(features_pca[idx_1, 0], features_pca[idx_1, 1], c='red', label='Fake (AI)', alpha=0.6)
    
    plt.title(f"PCA Visualization of Image Features (ResNet50)\nDataset: {os.path.basename(data_dir)}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_file, dpi=300)
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/analyze_pca.py <data_dir>")
        sys.exit(1)
        
    run_pca_analysis(sys.argv[1])
