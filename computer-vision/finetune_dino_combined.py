import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from pytorch_metric_learning import losses, miners
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import itertools
import json
import timm
import shutil

# Configuration
USE_BASE_MODEL = True  # Set to True for smarter features (Base), False for faster inference (Small)
IMG_SIZE = 448        # Match the new high-res standard

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

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on {device}")

# 1. Transforms (Now using SquarePadResize and IMG_SIZE)
studio_transform = T.Compose([
    SquarePadResize(IMG_SIZE, fill=255),
    T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.0)), 
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

shelf_transform = T.Compose([
    SquarePadResize(IMG_SIZE, fill=255),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Custom Combined Dataset
class CombinedProductDataset(Dataset):
    def __init__(self, studio_dir, shelf_dir, studio_transform, shelf_transform):
        self.samples = []
        self.classes = set()
        
        studio_path = Path(studio_dir)
        shelf_path = Path(shelf_dir)
        
        if studio_path.exists():
            self.classes.update([d.name for d in studio_path.iterdir() if d.is_dir() and not d.name.startswith('.')])
        if shelf_path.exists():
            self.classes.update([d.name for d in shelf_path.iterdir() if d.is_dir() and not d.name.startswith('.')])
            
        self.classes = sorted(list(self.classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        if studio_path.exists():
            for prod_dir in studio_path.iterdir():
                if prod_dir.is_dir() and not prod_dir.name.startswith('.'):
                    label = self.class_to_idx[prod_dir.name]
                    for img_path in prod_dir.glob("*.*"):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            self.samples.append((str(img_path), label, studio_transform))
                            
        if shelf_path.exists():
            for prod_dir in shelf_path.iterdir():
                if prod_dir.is_dir() and not prod_dir.name.startswith('.'):
                    label = self.class_to_idx[prod_dir.name]
                    for img_path in prod_dir.glob("*.*"):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            self.samples.append((str(img_path), label, shelf_transform))

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label, transform = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        return img_tensor, label

print("Loading combined dataset (Studio + Real Shelf)...")
dataset = CombinedProductDataset(
    studio_dir="data/NM_NGD_product_images", 
    shelf_dir="data/real_shelf_crops",
    studio_transform=studio_transform,
    shelf_transform=shelf_transform
)
print(f"Loaded {len(dataset)} total images across {len(dataset.classes)} unique products.")

# 3. Model Definition
class DINOFinetuner(nn.Module):
    def __init__(self, model_name, embed_dim):
        super().__init__()
        # Explicitly passing img_size=IMG_SIZE for positional embedding interpolation
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=IMG_SIZE)
        # Freeze the first 9 blocks to preserve foundational features

        for i, block in enumerate(self.backbone.blocks):
            if i < 9:
                for param in block.parameters():
                    param.requires_grad = False
                    
        # We fine-tune the native embeddings directly (768-dim for Base)
        self.head = nn.Identity()

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.head(features)
        return nn.functional.normalize(embeddings, p=2, dim=1)

# 4. Training Function for a single trial
def train_trial(trial_name, output_dir, batch_size, lr, margin, epochs, model_name, embed_dim):
    print(f"\n{'='*50}")
    print(f"Starting Trial: {trial_name}")
    print(f"Params: Batch={batch_size}, LR={lr}, Margin={margin}, Epochs={epochs}")
    print(f"Architecture: {model_name}")
    print(f"{'='*50}")
    
    trial_dir = output_dir / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # DataLoader
    actual_batch = min(batch_size, len(dataset))
    dataloader = DataLoader(dataset, batch_size=actual_batch, shuffle=True, num_workers=4, drop_last=True)
    
    # Init Model & Optimizer
    model = DINOFinetuner(model_name, embed_dim).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    miner = miners.MultiSimilarityMiner()
    loss_func = losses.TripletMarginLoss(margin=margin)
    
    best_loss = float('inf')
    best_model_path = trial_dir / "best_model.pt"
    
    trial_history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model(images)
            
            hard_pairs = miner(embeddings, labels)
            loss = loss_func(embeddings, labels, hard_pairs)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        
        trial_history.append({"epoch": epoch, "loss": avg_loss})
        
        # Save best model state dict
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"New best loss: {best_loss:.4f}. Saving weights...")
            torch.save(model.state_dict(), best_model_path)
            
    # Save trial metadata
    with open(trial_dir / "results.json", "w") as f:
        json.dump({
            "params": {"batch_size": batch_size, "lr": lr, "margin": margin, "epochs": epochs},
            "model_name": model_name,
            "best_loss": best_loss,
            "history": trial_history
        }, f, indent=2)
        
    return best_loss, best_model_path

# 5. Hyperparameter Tuning Loop
if USE_BASE_MODEL:
    model_name = 'vit_base_patch14_dinov2.lvd142m'
    embed_dim = 768
    print("Configured for DINOv2 BASE model tuning...")
else:
    model_name = 'vit_small_patch14_dinov2.lvd142m'
    embed_dim = 384
    print("Configured for DINOv2 SMALL model tuning...")

PARAM_GRID = {
    "batch_size": [64, 128],
    "lr": [1e-5, 5e-6],
    "margin": [0.1, 0.2],
    "epochs": [20]
}

base_output_dir = Path("runs/dino_tuning")
base_output_dir.mkdir(parents=True, exist_ok=True)

keys, values = zip(*PARAM_GRID.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"\nStarting Grid Search with {len(combinations)} combinations...")

best_overall_loss = float('inf')
best_overall_model = None
best_overall_params = None

for i, params in enumerate(combinations):
    trial_name = f"trial_{i+1}"
    
    try:
        final_loss, model_path = train_trial(
            trial_name=trial_name,
            output_dir=base_output_dir,
            model_name=model_name,
            embed_dim=embed_dim,
            **params
        )
        
        if final_loss < best_overall_loss:
            best_overall_loss = final_loss
            best_overall_model = model_path
            best_overall_params = params
            
    except RuntimeError as e:
        if "Out of memory" in str(e):
            print(f"OOM Error on {trial_name} with batch_size {params['batch_size']}. Skipping...")
            torch.cuda.empty_cache()
            continue
        else:
            raise e

print(f"\n{'='*50}")
print("TUNING COMPLETE!")
print(f"Best Loss Achieved: {best_overall_loss:.4f}")
print(f"Best Parameters: {best_overall_params}")
print(f"Best Model Weights saved at: {best_overall_model}")
print(f"{'='*50}")

final_model_dest = Path("best_dino.pt")
if best_overall_model and best_overall_model.exists():
    shutil.copy(best_overall_model, final_model_dest)
    print(f"\nCopied the absolute best weights to: {final_model_dest}")
