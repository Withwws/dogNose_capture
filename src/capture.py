import cv2
import torch
from model import DETR
import albumentations as A
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from utils.rich_handlers import DetectionHandler
import numpy as np

def detect_and_annotate(image):
    """
    Receives an image (numpy array), runs detection, draws results, and returns the processed image.
    """
    # logger = get_logger("image_process")
    detection_handler = DetectionHandler()

    # Define transforms
    transforms = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2()
    ])

    # Load model and resources
    model = DETR(num_classes=3)
    model.eval()
    state_dict = torch.load('checkpoints/99_model.pt', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    CLASSES = get_classes()
    COLORS = get_colors()
    
    # Transform and run inference
    transformed = transforms(image=image)
    result = model(torch.unsqueeze(transformed['image'], dim=0))
    probabilities = result['pred_logits'].softmax(-1)[:,:,:-1]
    max_probs, max_classes = probabilities.max(-1)
    keep_mask = max_probs > 0.8

    batch_indices, query_indices = torch.where(keep_mask)
    bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (image.shape[1], image.shape[0]))
    classes = max_classes[batch_indices, query_indices]
    probas = max_probs[batch_indices, query_indices]
    print("Model and resources loaded.")
    detections = []
    output_image = image.copy()
    print("Model and resources loaded22.")

    for bclass, bprob, bbox in zip(classes, probas, bboxes):
        bclass_idx = int(bclass.detach().cpu().numpy())
        bprob_val = float(bprob.detach().cpu().numpy())
        x1, y1, x2, y2 = bbox.detach().cpu().numpy()
        color = tuple(int(c) for c in COLORS[bclass_idx])

        detections.append({
            'class': CLASSES[bclass_idx],
            'confidence': bprob_val,
            'bbox': [float(x1), float(y1), float(x2), float(y2)]
        })

        # Draw bounding boxes and label
        output_image = cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        frame_text = f"{CLASSES[bclass_idx]} - {round(bprob_val, 4)}"
        output_image = cv2.rectangle(output_image, (int(x1), int(y1)-30), (int(x1)+200, int(y1)), color, -1)
        output_image = cv2.putText(output_image, frame_text, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    detection_handler.log_detections(detections, frame_id=0)
    print("Detections logged.")
    return output_image

# Example usage:
def runModel(image_path):
    # image_path = "your_image.jpg"
    input_image = cv2.imread(image_path)
    if input_image is None:
        print("Failed to load image.")
        return None
    else:
        print("Image loaded successfully.")
        processed_image = detect_and_annotate(input_image)
        output_path = "processed_image.jpg"
        cv2.imwrite(output_path, processed_image)
        return output_path