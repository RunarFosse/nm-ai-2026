import json
import cv2
import os
from pathlib import Path
from tqdm import tqdm

print("Setting up paths...")
base_dir = Path("data/train")
json_path = base_dir / "annotations.json"
images_dir = base_dir / "images"
metadata_path = Path("data/NM_NGD_product_images/metadata.json")
output_dir = Path("data/real_shelf_crops")

# 1. Load Metadata to map names to codes
print("Loading metadata...")
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Create a mapping of lowercase product name to product code
name_to_code = {p["product_name"].strip().lower(): p["product_code"] for p in metadata.get("products", [])}

# 2. Load COCO Annotations
print("Loading COCO annotations...")
with open(json_path, "r") as f:
    coco_data = json.load(f)

# Map category_id to product_code
category_to_code = {}
for cat in coco_data["categories"]:
    name = cat["name"].strip().lower()
    if name in name_to_code:
        category_to_code[cat["id"]] = name_to_code[name]

print(f"Successfully mapped {len(category_to_code)} out of {len(coco_data['categories'])} categories to product codes.")

# Map image_id to filename
image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

# Group annotations by image
annotations_by_image = {}
for ann in coco_data["annotations"]:
    img_id = ann["image_id"]
    if img_id not in annotations_by_image:
        annotations_by_image[img_id] = []
    annotations_by_image[img_id].append(ann)

# 3. Extract Crops
print("Starting crop extraction...")
extracted_count = 0
skipped_count = 0

for img_id, filename in tqdm(image_id_to_filename.items(), desc="Processing Images"):
    img_path = images_dir / filename
    if not img_path.exists():
        continue
        
    anns = annotations_by_image.get(img_id, [])
    if not anns:
        continue
        
    # Lazy load image only if there are valid annotations
    img = None 
    
    for ann in anns:
        cat_id = ann["category_id"]
        
        # If we don't know the product code for this category, skip it
        if cat_id not in category_to_code:
            skipped_count += 1
            continue
            
        product_code = category_to_code[cat_id]
        
        # Create output directory for this product code
        prod_dir = output_dir / product_code
        prod_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO bbox format: [x_min, y_min, width, height]
        x, y, w, h = map(int, ann["bbox"])
        
        # Load image if not loaded yet
        if img is None:
            img = cv2.imread(str(img_path))
            
        # Crop
        crop = img[y:y+h, x:x+w]
        
        # Ensure crop is valid (not empty or out of bounds)
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            skipped_count += 1
            continue
            
        # Save crop
        crop_filename = f"crop_{img_id}_{ann['id']}.jpg"
        cv2.imwrite(str(prod_dir / crop_filename), crop)
        extracted_count += 1

print(f"\nExtraction complete!")
print(f"Successfully extracted {extracted_count} real shelf crops.")
print(f"Skipped {skipped_count} crops (unknown category or invalid box).")
print(f"Saved all crops to {output_dir.absolute()}")
