import argparse
import json
from pathlib import Path
import torch
from model import PredictionModel
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
 
    # Initialize our pipeline
    print("Initializing Prediction Model...")
    model = PredictionModel()
    
    predictions = []
    
    # Process each image
    for img in sorted(Path(args.input).iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
            
        print(f"Processing {img.name}...")
        
        # model.forward returns a list of dictionaries formatted for the competition
        image_predictions = model(str(img))
        predictions.extend(image_predictions)
 
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Successfully saved {len(predictions)} predictions to {args.output}")
 
if __name__ == "__main__":
    main()