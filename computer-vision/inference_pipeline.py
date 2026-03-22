import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys
import json

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

def main(image_path="data/train/images/img_00019.jpg", yolo_model_path="best.pt", embeddings_path="studio_embeddings.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Studio Embeddings
    print("Loading studio embeddings...")
    if not Path(embeddings_path).exists():
        print(f"Error: {embeddings_path} not found. Run extract_studio_embeddings.py first.")
        return
    
    studio_data = torch.load(embeddings_path, map_location=device)
    studio_embeddings = studio_data["embeddings"].to(device)
    studio_ids = np.array(studio_data["product_ids"])
    studio_names = np.array(studio_data.get("product_names", ["Unknown"] * len(studio_ids)))
    studio_cat_ids = np.array(studio_data.get("category_ids", [-1] * len(studio_ids)))

    # 2. Load Models
    print(f"Loading YOLO model from {yolo_model_path}...")
    try:
        yolo_model = YOLO(yolo_model_path)
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return
    
    # Configuration
    USE_BASE_MODEL = True # MUST match your extract_studio_embeddings setting!
    dino_path = Path("best_dino.pt")
    IMG_SIZE = 448

    print("Loading DINOv2 model via timm...")
    import timm
    import torch.nn as nn
    
    if USE_BASE_MODEL:
        model_name = 'vit_base_patch14_dinov2.lvd142m'
        embed_dim = 768
    else:
        model_name = 'vit_small_patch14_dinov2.lvd142m'
        embed_dim = 384
    
    class DINOFinetuner(nn.Module):
        def __init__(self, model_name, embed_dim):
            super().__init__()
            self.backbone = timm.create_model(model_name, pretrained=not dino_path.exists(), num_classes=0, img_size=IMG_SIZE)
            # Always output the native embedding dimension (768 for Base)
            self.head = nn.Identity()


        def forward(self, x):
            features = self.backbone(x)
            embeddings = self.head(features)
            return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    dino_model = DINOFinetuner(model_name, embed_dim)
    
    if dino_path.exists():
        print(f"Loading custom fine-tuned weights from {dino_path} (OFFLINE)...")
        dino_model.load_state_dict(torch.load(dino_path, map_location=device), strict=False)
    else:
        print("Using PURE zero-shot foundational features.")
    
    dino_model.to(device)
    dino_model.eval()

    dino_transform = T.Compose([
        SquarePadResize(IMG_SIZE, fill=255),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Stage 1: Detection
    print(f"Running Stage 1 (YOLO detection) on {image_path}...")
    results = yolo_model(image_path, conf=0.1, iou=0.5, max_det=1000, agnostic_nms=True, imgsz=1536, verbose=False)
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy() # YOLO confidence scores
    print(f"Found {len(boxes)} potential products.")

    # 4. Stage 2: Extraction & Matching
    print("Running Stage 2 (DINOv2 identification)...")
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    predictions = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        box_conf = confs[i]
        crop = img_rgb[y1:y2, x1:x2]
        
        # Avoid empty crops (e.g. out of bounds)
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
            
        crop_pil = Image.fromarray(crop)
        crop_tensor = dino_transform(crop_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb = dino_model(crop_tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            
        # Match using Cosine Similarity
        similarities = torch.matmul(studio_embeddings, emb.T).squeeze()
        best_idx = torch.argmax(similarities).item()
        best_score = similarities[best_idx].item() # DINO similarity score
        best_product_id = studio_ids[best_idx]
        best_product_name = studio_names[best_idx]
        best_category_id = int(studio_cat_ids[best_idx])
        
        # Final Score: Combine YOLO objectness with DINO similarity
        # P(Product) = P(is_object) * P(is_class | object)
        final_score = float(box_conf * best_score)
        
        # Extract numeric ID from filename: "img_00019" -> 19
        try:
            image_id = int(Path(image_path).stem.split('_')[-1])
        except:
            image_id = 0

        predictions.append({
            "image_id": image_id,
            "category_id": best_category_id,
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], # [x, y, w, h]
            "score": round(final_score, 3),
            "product_name": best_product_name # Useful for manual checking
        })
        
    # Print the top 10 matches
    print("\n--- Detection Results ---")
    for i, p in enumerate(predictions[:10]):
        print(f"Crop {i+1}: {p['product_name']} (Cat: {p['category_id']}) [Final Score: {p['score']}]")
    
    # Save to JSON
    with open("submission.json", "w") as f:
        # We strip the "product_name" for the final competition submission
        submission_output = [
            {k: v for k, v in p.items() if k != "product_name"} 
            for p in predictions
        ]
        json.dump(submission_output, f, indent=2)
    print("\nSaved competition-formatted results to submission.json")

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/train/images/img_00019.jpg"
    main(img_path)
