import json
import os
import random
import shutil
from pathlib import Path

# Paths
base_dir = Path("data/train")
json_path = base_dir / "annotations.json"
images_dir = base_dir / "images"
labels_dir = base_dir / "labels"

# New Split Folders
split_dir = base_dir.parent / "norgesgruppen_dataset"
for split in ["train", "val"]:
    (split_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (split_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# Create labels directory temporarily
labels_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading COCO annotations from {json_path}...")
with open(json_path, "r") as f:
    coco_data = json.load(f)

# Create a mapping of image_id to image info (width, height, file_name)
images_info = {
    img["id"]: {"file_name": img["file_name"], "width": img["width"], "height": img["height"]}
    for img in coco_data.get("images", [])
}

# Group annotations by image_id
annotations_by_image = {}
for ann in coco_data.get("annotations", []):
    img_id = ann["image_id"]
    if img_id not in annotations_by_image:
        annotations_by_image[img_id] = []
    annotations_by_image[img_id].append(ann)

print("Converting bounding boxes to YOLO format...")
converted_count = 0

for img_id, img_info in images_info.items():
    file_name = img_info["file_name"]
    img_w = img_info["width"]
    img_h = img_info["height"]
    
    # Text file for this image
    txt_filename = Path(file_name).stem + ".txt"
    txt_path = labels_dir / txt_filename
    
    # Get annotations for this image
    anns = annotations_by_image.get(img_id, [])
    
    yolo_lines = []
    for ann in anns:
        # COCO bbox format: [x_min, y_min, width, height]
        bbox = ann["bbox"]
        x_min, y_min, w, h = bbox
        
        # Calculate YOLO format: x_center, y_center, width, height (normalized)
        x_center = (x_min + w / 2) / img_w
        y_center = (y_min + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        
        # Ensure values are between 0 and 1
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_w = max(0.0, min(1.0, norm_w))
        norm_h = max(0.0, min(1.0, norm_h))
        
        # We set class_id to 0 for a generic "product" detector
        class_id = 0 
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
    
    # Write to file even if empty (YOLO needs empty text files for images with no objects)
    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines))
    
    converted_count += 1

print(f"Successfully converted {converted_count} images to YOLO format.")

print("Splitting dataset into 80% train and 20% val...")
all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
# Sort and then shuffle with a fixed seed for reproducibility
all_images.sort()
random.seed(42)
random.shuffle(all_images)

split_idx = int(len(all_images) * 0.8)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

def move_files(file_list, split_name):
    for img_name in file_list:
        # Copy Image
        shutil.copy(images_dir / img_name, split_dir / "images" / split_name / img_name)
        
        # Copy corresponding Label (.txt)
        label_name = Path(img_name).stem + ".txt"
        if (labels_dir / label_name).exists():
            shutil.copy(labels_dir / label_name, split_dir / "labels" / split_name / label_name)

print(f"Moving {len(train_images)} images to Train...")
move_files(train_images, "train")

print(f"Moving {len(val_images)} images to Val...")
move_files(val_images, "val")

# Create the YAML file required for YOLO training
yaml_path = base_dir.parent / "norgesgruppen.yaml"
yaml_content = f"""path: {split_dir.absolute()}
train: images/train
val: images/val

names:
  0: product
"""
with open(yaml_path, "w") as f:
    f.write(yaml_content)
print(f"Created dataset configuration file at {yaml_path}")

