import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import timm
import torch.nn as nn

class SquarePadResize(object):
    """
    Pads an image to a square (preserving aspect ratio) using white/black bars,
    then resizes it to the target size. This prevents "squishing" retail products.
    """
    def __init__(self, target_size, fill=255):
        self.target_size = target_size if isinstance(target_size, tuple) else (target_size, target_size)
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        img = TF.pad(img, padding, fill=self.fill, padding_mode='constant')
        return TF.resize(img, self.target_size)
        
    def __repr__(self):
        return self.__class__.__name__ + f'(target_size={self.target_size}, fill={self.fill})'

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
USE_BASE_MODEL = True  # Set to True for smarter features (Base), False for faster inference (Small)
weights_path = Path("best_dino.pt")
IMG_SIZE = 448 # High resolution for fine-grained text reading

# 1. Load DINOv2 Model via timm
if USE_BASE_MODEL:
    model_name = 'vit_base_patch14_dinov2.lvd142m'
    embed_dim = 768
    print("Using DINOv2 BASE model (Higher Accuracy)...")
else:
    model_name = 'vit_small_patch14_dinov2.lvd142m'
    embed_dim = 384
    print("Using DINOv2 SMALL model (Higher Speed)...")

class DINOFinetuner(nn.Module):
    def __init__(self, model_name, embed_dim, use_head=True):
        super().__init__()
        # Load from timm with IMG_SIZE
        self.backbone = timm.create_model(model_name, pretrained=not weights_path.exists(), num_classes=0, img_size=IMG_SIZE)
        # Always use the native embedding dimension (768 for Base)
        self.head = nn.Identity()

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.head(features)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

# Only use the projection head if we have trained weights!
use_custom_head = weights_path.exists()
model = DINOFinetuner(model_name, embed_dim, use_head=use_custom_head)

if weights_path.exists():
    print(f"Loading custom fine-tuned weights from {weights_path} (OFFLINE)...")
    # Note: Ensure the weights match the USE_BASE_MODEL setting!
    # strict=False allows timm to interpolate the positional embeddings from 224 to 448 automatically!
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
else:
    print("No fine-tuned weights found. Using raw zero-shot foundational features (No Head).")

model.to(device)
model.eval()

# 2. DINOv2 Transforms
transform = T.Compose([
    SquarePadResize(IMG_SIZE, fill=255),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Path to studio images and metadata
studio_base_path = Path("data/NM_NGD_product_images")
metadata_path = studio_base_path / "metadata.json"
annotations_path = Path("data/train/annotations.json")
output_path = Path("studio_embeddings.pt")

# Load category and name mapping
print("Loading product metadata and COCO categories...")
code_to_name = {}
name_to_id = {}
code_to_cat_id = {}

try:
    # Get Name -> Category ID from annotations.json
    with open(annotations_path, 'r') as f:
        coco = json.load(f)
        for cat in coco.get('categories', []):
            name_to_id[cat['name'].strip().lower()] = cat['id']
            
    # Get Code -> Name from metadata.json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        for prod in metadata.get('products', []):
            code = prod['product_code']
            name = prod['product_name'].strip().lower()
            code_to_name[code] = prod['product_name']
            # Link them
            if name in name_to_id:
                code_to_cat_id[code] = name_to_id[name]
                
except Exception as e:
    print(f"Warning during metadata loading: {e}")

# 4. Extraction Loop
studio_data = {
    "embeddings": [],
    "product_ids": [],
    "product_names": [],
    "category_ids": [],
    "filenames": []
}

# Get all product directories (ignore hidden files/folders)
product_dirs = [d for d in studio_base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

print(f"Found {len(product_dirs)} products. Starting multi-view extraction...")

for product_dir in tqdm(product_dirs):
    product_code = product_dir.name
    product_name = code_to_name.get(product_code, "unknown_product")
    category_id = code_to_cat_id.get(product_code, 355) # 355 is the COCO id for unknown_product
    
    # Get all images in this product folder
    image_files = [f for f in product_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    for img_file in image_files:
        try:
            # Load image
            img = Image.open(img_file).convert('RGB')
            w, h = img.size
            
            crops = {
                "full": img,
                "top": TF.crop(img, top=0, left=0, height=h//2, width=w),
                "bottom": TF.crop(img, top=h//2, left=0, height=h//2, width=w),
                "left": TF.crop(img, top=0, left=0, height=h, width=w//2),
                "right": TF.crop(img, top=0, left=w//2, height=h, width=w//2),
            }
            
            for crop_name, crop_img in crops.items():
                img_tensor = transform(crop_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = model(img_tensor)
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                
                # Store
                studio_data["embeddings"].append(embedding.cpu())
                studio_data["product_ids"].append(product_code)
                studio_data["product_names"].append(product_name)
                studio_data["category_ids"].append(category_id)
                studio_data["filenames"].append(f"{img_file.name}_{crop_name}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

if studio_data["embeddings"]:
    studio_data["embeddings"] = torch.cat(studio_data["embeddings"], dim=0)
    print(f"Saving {len(studio_data['product_ids'])} multi-view embeddings to {output_path}...")
    torch.save(studio_data, output_path)
    print("Done!")


