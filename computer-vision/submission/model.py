import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import onnxruntime as ort
import torchvision

from utils import DINOFinetuner, SquarePadResize

class PredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = 'vit_base_patch14_dinov2.lvd142m'
        embed_dim = 768
        
        base_dir = Path(__file__).parent
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Initialize ONNX Runtime Detector (YOLOv12)
        onnx_path = str(base_dir / "best_finetune.onnx")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self._device.type == 'cuda' else ['CPUExecutionProvider']
        print(f"Loading ONNX Detector from {onnx_path}...")
        self._detector_session = ort.InferenceSession(onnx_path, providers=providers)
        self._detector_input_name = self._detector_session.get_inputs()[0].name
        
        # 2. Load DINO (Identifier)
        weights_path = base_dir / 'best_dino.pt'
        self._identifier = DINOFinetuner(model_name, embed_dim)
        if weights_path.exists():
            print(f"Loading custom DINO weights from {weights_path}...")
            self._identifier.load_state_dict(torch.load(str(weights_path), map_location=self._device), strict=False)
        
        self._identifier.to(self._device)
        if self._device.type == 'cuda':
            self._identifier.half()
        self._identifier.eval()

        # 3. Transforms
        self._identifier_transforms = T.Compose([
            SquarePadResize(448, fill=255),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 4. Search Index
        print("Loading search index...")
        self._products = torch.load(str(base_dir / "studio_embeddings.pt"), map_location=self._device)
        if isinstance(self._products['embeddings'], torch.Tensor) and self._device.type == 'cuda':
            self._products['embeddings'] = self._products['embeddings'].half()

    def _letterbox(self, img, new_shape=(1088, 1088)):
        # Resizes and pads image while preserving aspect ratio
        shape = img.shape[:2] # current [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = dw / 2, dh / 2 # divide padding into 2 sides
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, (dw, dh)

    def forward(self, image_path):
        # 1. Stage 1: Manual ONNX Inference
        img0 = cv2.imread(str(image_path))
        if img0 is None: return []
        h0, w0 = img0.shape[:2]
        
        # Pre-process
        img, ratio, (dw, dh) = self._letterbox(img0)
        img = img.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        img = img[None, ...] # Add batch dimension
        
        # Run ONNX
        outputs = self._detector_session.run(None, {self._detector_input_name: img})
        output = torch.from_numpy(outputs[0]) # [1, 5, 48384]
        
        # Post-process (Decoding + NMS)
        output = output[0].transpose(0, 1) # [N, 5]
        boxes = output[:, :4] # [cx, cy, w, h]
        scores = output[:, 4]
        
        mask = scores > 0.1
        boxes, scores = boxes[mask], scores[mask]
        if len(boxes) == 0: return []
        
        # Convert to x1y1x2y2
        x1 = (boxes[:, 0] - boxes[:, 2] / 2 - dw) / ratio
        y1 = (boxes[:, 1] - boxes[:, 3] / 2 - dh) / ratio
        x2 = (boxes[:, 0] + boxes[:, 2] / 2 - dw) / ratio
        y2 = (boxes[:, 1] + boxes[:, 3] / 2 - dh) / ratio
        boxes_scaled = torch.stack([x1, y1, x2, y2], dim=1)
        
        indices = torchvision.ops.nms(boxes_scaled, scores, 0.5)
        boxes_scaled, scores = boxes_scaled[indices], scores[indices]
        
        # 2. Stage 2: Identification
        image_id = int(Path(image_path).stem.split('_')[-1])
        img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        
        crop_tensors = []
        valid_boxes = []
        valid_scores = []
        
        for i in range(len(boxes_scaled)):
            bx = boxes_scaled[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, bx)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w0, x2), min(h0, y2)
            
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            crop_pil = Image.fromarray(crop)
            crop_tensors.append(self._identifier_transforms(crop_pil))
            valid_boxes.append([float(x1), float(y1), float(x2-x1), float(y2-y1)])
            valid_scores.append(float(scores[i]))

        if not crop_tensors: return []
        
        # Mini-batch inference for DINO
        batch_size = 64
        all_embeddings = []
        with torch.no_grad():
            for s in range(0, len(crop_tensors), batch_size):
                b = torch.stack(crop_tensors[s:s+batch_size]).to(self._device)
                if self._device.type == 'cuda': b = b.half()
                all_embeddings.append(self._identifier(b))
        embeddings = torch.cat(all_embeddings, dim=0)
        
        # 3. Vector Matching
        gallery_embs = self._products['embeddings'].to(self._device)
        cat_ids = np.array(self._products['category_ids'])
        sims = torch.matmul(embeddings, gallery_embs.T)
        best_sims, best_idx = torch.max(sims, dim=1)
        
        # 4. Final Format
        predictions = []
        for i in range(len(valid_boxes)):
            final_score = valid_scores[i] * best_sims[i].item()
            predictions.append({
                "image_id": image_id,
                "category_id": int(cat_ids[best_idx[i]]),
                "bbox": valid_boxes[i],
                "score": round(final_score, 3)
            })
        return predictions

    def _get_embedding_from_crop(self, image_np):
        if isinstance(image_np, np.ndarray): img_pil = Image.fromarray(image_np)
        else: img_pil = image_np
        img_t = self._identifier_transforms(img_pil).unsqueeze(0).to(self._device)
        if self._device.type == 'cuda': img_t = img_t.half()
        with torch.no_grad():
            emb = self._identifier(img_t)
        return F.normalize(emb, p=2, dim=1).cpu().numpy()