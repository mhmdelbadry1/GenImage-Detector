import torch
import torch.nn as nn
import timm

def get_model(model_name, num_classes=2, pretrained=True):
    """
    Factory function to get models.
    
    Supported models:
    - 'resnet50'
    - 'xception'
    - 'vit_base_patch16_224'
    """
    
    if model_name == 'resnet50':
        # Baseline ResNet50
        model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)
        
    elif model_name == 'xception':
        # Xception (known for good deepfake detection)
        model = timm.create_model('xception', pretrained=pretrained, num_classes=num_classes)
        
    elif model_name == 'vit':
        # Vision Transformer
        # vit_base_patch16_224 is a standard ViT
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from: resnet50, xception, vit")
        
    return model

if __name__ == "__main__":
    # Test block
    try:
        m = get_model('resnet50')
        print("Successfully loaded ResNet50")
        m = get_model('xception')
        print("Successfully loaded Xception")
        m = get_model('vit')
        print("Successfully loaded ViT")
    except Exception as e:
        print(f"Error loading models: {e}")
