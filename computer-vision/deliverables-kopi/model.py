from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image

from utils import DINOFinetuner, SquarePadResize

class PredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = 'vit_base_patch14_dinov2.lvd142m'
        embed_dim = 768
        
        base_dir = Path(__file__).parent
        weights_path = base_dir / 'best_dino.pt'
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Load YOLO (Detector) in Half Precision
        self._detector = YOLO(str(base_dir / "best_finetune.pt"))
        
        # 2. Load DINO (Identifier)
        self._identifier = DINOFinetuner(model_name, embed_dim)
        if weights_path.exists():
            print(f"Loading custom fine-tuned weights from {weights_path} (OFFLINE MODE)...")
            self._identifier.load_state_dict(torch.load(str(weights_path), map_location=self._device), strict=False)
        else:
            print("Using PURE zero-shot foundational features (ONLINE MODE).")

        # Move models to GPU and convert to FP16 for speed
        self._detector.to(self._device)
        self._identifier.to(self._device)
        
        if self._device.type == 'cuda':
            print("Optimizing models for FP16 (Half Precision)...")
            self._identifier.half()
            # YOLO handles .half() internally during inference if half=True is passed

        self._detector.eval()
        self._identifier.eval()

        # 3. DINOv2 Transforms (High-Res 448x448 + Aspect Ratio Padding)
        self._identifier_transforms = T.Compose([
            SquarePadResize(448, fill=255),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 4. Load the pre-computed studio embeddings (and ensure they are Half Precision)
        print("Loading search index...")
        self._products = torch.load(str(base_dir / "studio_embeddings.pt"), map_location=self._device)
        if isinstance(self._products['embeddings'], torch.Tensor) and self._device.type == 'cuda':
            self._products['embeddings'] = self._products['embeddings'].half()

    def forward(self, image_path):
        """
        Performs end-to-end detection and identification on a single image.
        Returns a list of dicts in the competition JSON format.
        """
        # 1. Stage 1: YOLO Detection
        results = self._detector(
            image_path, 
            conf=0.1, 
            iou=0.5, 
            max_det=1000, 
            agnostic_nms=True, 
            imgsz=1536, 
            verbose=False
        )
        result = results[0]
        
        if len(result.boxes) == 0:
            return []

        try:
            image_id = int(Path(image_path).stem.split('_')[-1])
        except:
            image_id = 0

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return []
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        
        # 2. Stage 2: Batch Embedding Extraction
        crop_tensors = []
        valid_box_indices = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
            
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            crop_pil = Image.fromarray(crop)
            crop_tensors.append(self._identifier_transforms(crop_pil))
            valid_box_indices.append(i)

        if not crop_tensors:
            return []

        # Stack into batch and convert to Half Precision
        batch_tensor = torch.stack(crop_tensors).to(self._device)
        if self._device.type == 'cuda':
            batch_tensor = batch_tensor.half()
        
        with torch.no_grad():
            embeddings = self._identifier(batch_tensor)
            
        # 3. Stage 3: Vector Search (Similarity Matching)
        gallery_embeddings = self._products['embeddings'].to(self._device)
        category_ids = np.array(self._products['category_ids'])
        
        # Matrix multiplication handles the similarity matching
        similarities = torch.matmul(embeddings, gallery_embeddings.T)
        best_scores, best_indices = torch.max(similarities, dim=1)
        
        # 4. Final Format Assembly
        predictions = []
        for i, box_idx in enumerate(valid_box_indices):
            best_idx = best_indices[i].item()
            best_sim = best_scores[i].item()
            yolo_conf = confs[box_idx]
            
            final_score = float(yolo_conf * best_sim)
            x1, y1, x2, y2 = boxes[box_idx]
            
            predictions.append({
                "image_id": image_id,
                "category_id": int(category_ids[best_idx]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": round(final_score, 3)
            })
            
        return predictions


    def _get_embedding_from_crop(self, image_np):
        # Convert numpy array (from OpenCV/YOLO) to PIL Image
        if isinstance(image_np, np.ndarray):
            img_pil = Image.fromarray(image_np)
        else:
            img_pil = image_np # Already PIL
        
        # Apply transforms and add batch dimension: [1, 3, 448, 448]
        img_tensor = self._identifier_transforms(img_pil).unsqueeze(0).to(self._device)
        if self._device.type == 'cuda':
            img_tensor = img_tensor.half()
    
        # Extract features without computing gradients
        with torch.no_grad():
            embedding = self._identifier(img_tensor)
        
        # Normalize the embedding (crucial for cosine similarity)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()