"""
Deepfake Detection Web Application
Runs all 3 models (SimpleCNN, ResNet50, SSP) on separate GPUs
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from flask import Flask, request, render_template, jsonify
import io
import base64

# Add paths
sys.path.insert(0, '/home/astro/deepleaning')
sys.path.insert(0, '/home/astro/deepleaning/SSP-AI-Generated-Image-Detection')

from models.simple_cnn import SimpleCNN
from models.resnet50_baseline import ResNet50Baseline
from networks.ssp import ssp
from utils.patch import patch_img

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Model configurations
MODELS = {
    'SimpleCNN': {
        'checkpoint': '/home/astro/deepleaning/experiments/exp1_simple_cnn/checkpoints/best_model.pth',
        'gpu': 0,
        'class': SimpleCNN
    },
    'ResNet50': {
        'checkpoint': '/home/astro/deepleaning/experiments/exp2_resnet50/checkpoints/best_model.pth',
        'gpu': 1,
        'class': ResNet50Baseline
    },
    'SSP': {
        'checkpoint': '/home/astro/deepleaning/ssp_logo_checkpoints/best.pth',
        'gpu': 2,
        'class': ssp
    }
}

# Global model storage
loaded_models = {}

def load_models():
    """Load all 3 models on different GPUs"""
    print("="*60)
    print("Loading Deep fake Detection Models...")
    print("="*60)
    
    for name, config in MODELS.items():
        try:
            device = torch.device(f'cuda:{config["gpu"]}')
            
            # Initialize model
            if name == 'SSP':
                model = config['class']()
            else:
                model = config['class'](num_classes=2)
            
            # Load checkpoint
            checkpoint = torch.load(config['checkpoint'], map_location=device)
            model.load_state_dict(checkpoint)
            model = model.to(device)
            model.eval()
            
            loaded_models[name] = {
                'model': model,
                'device': device
            }
            
            print(f"✓ {name:12} loaded on GPU {config['gpu']}")
            
        except Exception as e:
            print(f"✗ {name:12} failed to load: {e}")
            loaded_models[name] = None
    
    print("="*60)
    print(f"Successfully loaded {len([m for m in loaded_models.values() if m is not None])}/3 models")
    print("="*60 + "\n")

def preprocess_standard(image):
    """Standard preprocessing for SimpleCNN and ResNet50"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def preprocess_ssp(image):
    """SSP preprocessing with patch extraction"""
    # Extract simplest patch
    # patch_img expects (img, patch_size, height)
    patch = patch_img(image, 32, 256)
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(patch).unsqueeze(0)

def predict_image(image_path):
    """Run inference on all 3 models"""
    results = {}
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Predict with SimpleCNN
    if loaded_models.get('SimpleCNN'):
        try:
            model_info = loaded_models['SimpleCNN']
            img_tensor = preprocess_standard(image).to(model_info['device'])
            
            with torch.no_grad():
                output = model_info['model'](img_tensor)
                probs = torch.softmax(output, dim=1)[0]
                fake_prob = probs[1].item()
                
                results['SimpleCNN'] = {
                    'prediction': 'AI-Generated' if fake_prob > 0.5 else 'Real',
                    'confidence': max(fake_prob, 1 - fake_prob) * 100,
                    'fake_probability': fake_prob * 100,
                    'real_probability': (1 - fake_prob) * 100
                }
        except Exception as e:
            results['SimpleCNN'] = {'error': str(e)}
    
    # Predict with ResNet50
    if loaded_models.get('ResNet50'):
        try:
            model_info = loaded_models['ResNet50']
            img_tensor = preprocess_standard(image).to(model_info['device'])
            
            with torch.no_grad():
                output = model_info['model'](img_tensor)
                probs = torch.softmax(output, dim=1)[0]
                fake_prob = probs[1].item()
                
                results['ResNet50'] = {
                    'prediction': 'AI-Generated' if fake_prob > 0.5 else 'Real',
                    'confidence': max(fake_prob, 1 - fake_prob) * 100,
                    'fake_probability': fake_prob * 100,
                    'real_probability': (1 - fake_prob) * 100
                }
        except Exception as e:
            results['ResNet50'] = {'error': str(e)}
    
    # Predict with SSP
    if loaded_models.get('SSP'):
        try:
            print(f"[SSP] Starting prediction...")
            model_info = loaded_models['SSP']
            
            print(f"[SSP] Preprocessing image with patch extraction...")
            img_tensor = preprocess_ssp(image).to(model_info['device'])
            print(f"[SSP] Patch tensor shape: {img_tensor.shape}")
            
            # Upscale to 256x256
            img_tensor = torch.nn.functional.interpolate(img_tensor, size=(256, 256), mode='bilinear')
            print(f"[SSP] Upscaled tensor shape: {img_tensor.shape}")
            
            with torch.no_grad():
                output = model_info['model'](img_tensor).item()
                print(f"[SSP] Model output (logit): {output}")
                prob = torch.sigmoid(torch.tensor(output)).item()
                print(f"[SSP] Sigmoid probability: {prob}")
                
                # SSP: nature=1, ai=0, so flip
                real_prob = prob
                fake_prob = 1 - prob
                
                results['SSP'] = {
                    'prediction': 'AI-Generated' if fake_prob > 0.5 else 'Real',
                    'confidence': max(fake_prob, real_prob) * 100,
                    'fake_probability': fake_prob * 100,
                    'real_probability': real_prob * 100
                }
                print(f"[SSP] Result: {results['SSP']}")
        except Exception as e:
            print(f"[SSP] ERROR: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            results['SSP'] = {'error': str(e)}
    
    return results

@app.route('/test')
def test():
    """Test page for debugging"""
    return render_template('test.html')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if file:
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_upload.jpg')
        file.save(filepath)
        
        # Run predictions
        results = predict_image(filepath)
        
        # Encode image for display
        with open(filepath, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        
        return jsonify({
            'results': results,
            'image': img_data
        })

if __name__ == '__main__':
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Load all models
    load_models()
    
    # Run Flask app
    print("\n🚀 Starting Deepfake Detection Web Server...")
    print("📍 Access at: http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
