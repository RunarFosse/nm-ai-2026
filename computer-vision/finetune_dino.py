import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, miners
from tqdm import tqdm
from pathlib import Path

# 1. Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5
MARGIN = 0.1 # Triplet loss margin

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on {device}")

# 2. Extreme Augmentations to simulate "Shelf Conditions"
# We want to take pristine 360 studio shots and make them look like blurry, poorly lit shelf crops
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.5, 1.0)), # Simulate partial occlusion/cropping
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Simulate fridge lighting
    T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.3), # Simulate out-of-focus smartphone shots
    T.RandomHorizontalFlip(),
    T.RandomRotation(15), # Products fall over
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset (ImageFolder automatically uses folder names as class labels)
dataset_path = "data/NM_NGD_product_images"
dataset = ImageFolder(root=dataset_path, transform=train_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
print(f"Loaded {len(dataset)} images across {len(dataset.classes)} products.")

# 3. Model Setup
print("Loading DINOv2...")
class DINOFinetuner(nn.Module):
    def __init__(self):
        super().__init__()
        # Load base model
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # Freeze the first 9 blocks to prevent catastrophic forgetting
        for i, block in enumerate(self.backbone.blocks):
            if i < 9:
                for param in block.parameters():
                    param.requires_grad = False
                    
        # Add a projection head (maps 384-dim to 256-dim for metric learning)
        self.head = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.head(features)
        # Always L2 normalize embeddings for cosine distance
        return nn.functional.normalize(embeddings, p=2, dim=1)

model = DINOFinetuner().to(device)

# 4. Optimizer and Metric Learning Loss
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# The Miner finds the "hardest" triplets in the batch to train on
miner = miners.MultiSimilarityMiner()
# The Loss function calculates the penalty
loss_func = losses.TripletMarginLoss(margin=MARGIN)

# 5. Training Loop
print("Starting Fine-tuning...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(images)
        
        # Find hard triplets and calculate loss
        hard_pairs = miner(embeddings, labels)
        loss = loss_func(embeddings, labels, hard_pairs)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    torch.save(model.state_dict(), f"dinov2_finetuned_epoch_{epoch}.pt")

print("Training complete! Best model saved.")
