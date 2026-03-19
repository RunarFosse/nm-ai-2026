import torch
from mmdet.apis import init_detector, inference_detector
import mmcv

def test_cascade():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # 1. Specify the model you want (Cascade R-CNN with ResNet-50)
    # MMDetection will automatically find this config in its library
    config_file = 'cascade-rcnn_r50_fpn_1x_coco.py'
    
    # 2. Provide the URL to the pre-trained weights
    # MMDetection will automatically download and cache this
    checkpoint_url = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'

    # 3. Initialize the model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model on {device}...")
    
    model = init_detector(config_file, checkpoint_url, device=device)
    print("Model loaded successfully!")

    # 4. (Optional) Run a dummy inference if you have an image ready
    # img = 'grocery_shelf.jpg' 
    # result = inference_detector(model, img)
    # print(result.pred_instances.bboxes) # The bounding boxes

if __name__ == "__main__":
    test_cascade()