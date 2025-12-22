import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GenImageDataset(Dataset):
    """
    Dataset class for GenImage.
    
    It recursively finds all images in the root directory.
    Labels are determined by keywords in the file path:
    - 'nature' -> 0 (Real)
    - 'ai'     -> 1 (Fake)
    
    Args:
        root_dir (str): Path to the specific dataset folder (e.g., 'data/BigGAN')
        transform (callable, optional): Optional transform to be applied on a sample.
        split (str): 'train' or 'val'. If None, uses all images found.
    """
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Standard ImageNet normalization if no transform is provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            
        print(f"Scanning {root_dir}...")
        
        # Supported extensions
        exts = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']
        all_files = []
        
        # Recursive search using os.walk to handle deep nesting
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.split('.')[-1] in exts:
                    full_path = os.path.join(root, file)
                    all_files.append(full_path)
        
        # Filter and assign labels
        for path in all_files:
            # Check for split if specified (assuming 'train' or 'val' is in path)
            # Since the structure is inconsistent (some have 'train/nature', some might be flat),
            # we try to honor the 'split' arg if the keyword exists in the path.
            # If the path DOES NOT contain 'train' or 'val' at all, we include it anyway 
            # (assuming naive dump), but if it DOES, we enforce the match.
            
            path_lower = path.lower()
            
            if split:
                if 'train' in path_lower and split != 'train':
                    continue
                if 'val' in path_lower and split != 'val':
                    continue
            
            label = None
            if '/nature/' in path_lower or '/0_real/' in path_lower:
                label = 0
            elif '/ai/' in path_lower or '/1_fake/' in path_lower:
                label = 1
            
            if label is not None:
                self.samples.append((path, label))
                
        print(f"Found {len(self.samples)} images for split='{split}' in {root_dir}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a dummy image or handle gracefully? 
            # For simplicity in this script, we'll try the next index or random
            # But here let's just crash or return black
            return self.__getitem__((idx + 1) % len(self))

if __name__ == "__main__":
    # Test block
    import sys
    
    if len(sys.argv) > 1:
        d = GenImageDataset(sys.argv[1], split=None)
        print(f"Sample 0 label: {d[0][1]}")
