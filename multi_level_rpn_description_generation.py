import os
import cv2
import numpy as np
import json
import argparse
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import easyocr 

def setup_predictor(use_cuda, confidence_threshold):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(confidence_threshold)  # Confidence threshold passed as argument
    
    # Set device to cuda or cpu based on use_cuda flag
    if use_cuda:
        cfg.MODEL.DEVICE = "cuda"
    else:
        cfg.MODEL.DEVICE = "cpu"
        
    return DefaultPredictor(cfg)

def setup_ocr():
    reader = easyocr.Reader(['ch_sim', 'en'])  # Supports Chinese and English
    return reader

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Failed to load image {image_path}, skipping.")
    return image

def generate_proposals(image, predictor):
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        return np.array([])
    outputs = predictor(image)
    boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    return boxes

def save_image(image, output_path):
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        print(f"Warning: Image is empty. Skipping save: {output_path}")
        return
    cv2.imwrite(output_path, image)

def is_contained(box, larger_box):
    x1, y1, x2, y2 = box
    lx1, ly1, lx2, ly2 = larger_box
    return lx1 <= x1 and ly1 <= y1 and lx2 >= x2 and ly2 >= y2

def generate_text_proposals(image, ocr, max_size=2048):
    h, w = image.shape[:2]

    # If the image is too large, resize to prevent OpenCV warpPerspective failure
    if max(h, w) > max_size:
        scale_factor = max_size / max(h, w)
        image_resized = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))
    else:
        image_resized = image

    result = ocr.readtext(image_resized)

    boxes = []
    for detection in result:
        bbox = detection[0]  # EasyOCR returns bbox as quadrilateral
        x_min, y_min = np.min(bbox, axis=0)
        x_max, y_max = np.max(bbox, axis=0)

        # Mapping coordinates back to the original image size
        if max(h, w) > max_size:
            x_min, y_min, x_max, y_max = (
                int(x_min / scale_factor),
                int(y_min / scale_factor),
                int(x_max / scale_factor),
                int(y_max / scale_factor),
            )

        boxes.append([x_min, y_min, x_max, y_max])

    return np.array(boxes)

def filter_boxes(boxes, level):
    if level >= 3:
        return boxes  # Keep all boxes for the third level
    
    filtered_boxes = []
    for i, box in enumerate(boxes):
        contained = any(is_contained(box, other_box) for j, other_box in enumerate(boxes) if i != j)
        if not contained:
            filtered_boxes.append(box)
    return np.array(filtered_boxes)

def process_image_recursive(image, output_dir, ocr, predictor, level=1, max_level=3, messages=None, image_id=1, box_counts=None):
    if messages is None:
        messages = []
    if box_counts is None:
        box_counts = {}
    if level > max_level or image is None or image.shape[0] == 0 or image.shape[1] == 0:
        return
    os.makedirs(output_dir, exist_ok=True)

    # Paper Algorithm 1: add root-image description d^(0) at level 1 before processing proposals
    if level == 1:
        original_path = os.path.join(output_dir, "original.jpg")
        root_prompt = "Describe the image in detail, highlighting key elements and their relationships. Keep it under 150 words."
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": original_path},
                {"type": "text", "text": root_prompt}
            ]
        })

    rpn_proposals  = generate_proposals(image, predictor)
    if level == 1:
        text_proposals = generate_text_proposals(image, ocr)
    else:
        text_proposals = np.array([])

    rpn_proposals  = filter_boxes(rpn_proposals, level)  # Filter contained boxes for the first two levels

    # **Fix boolean check**
    if rpn_proposals.size == 0 and text_proposals.size == 0 and level == 1:
        print(f"⚠️ {output_dir} did not detect RPN or OCR regions, skipping")
        return  # Skip image to avoid errors later

    if rpn_proposals.size == 0:  # If RPN is empty, use text proposals
        proposals = text_proposals
    elif text_proposals.size == 0:  # If text proposals are empty, use RPN boxes
        proposals = rpn_proposals
    else:  # If both are available, merge them
        proposals = np.vstack((rpn_proposals, text_proposals))
    
    box_counts[level] = box_counts.get(level, 0) + len(proposals)  # Accumulate box counts across recursive branches
    
    if len(proposals) == 0:
        return
    
    for i, box in enumerate(proposals):
        x1, y1, x2, y2 = map(int, box)
        sub_image = image[y1:y2, x1:x2]
        
        if sub_image is None or sub_image.shape[0] == 0 or sub_image.shape[1] == 0:
            continue  # Skip empty images
        
        sub_image_path = os.path.join(output_dir, f"{i + 1}.jpg")
        save_image(sub_image, sub_image_path)
        
        if level == 1:
            prompt_text = "Describe the image in detail, highlighting key elements and their relationships. Keep it under 150 words."
        elif level == 2:
            prompt_text = "Please list the names of all objects in the image, separated by commas, and then describe the image. Limit the description to 30 words."
        else:
            prompt_text = "What is this object?"
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": sub_image_path},
                {"type": "text", "text": prompt_text}
            ]
        })
        
        sub_image_dir = os.path.join(output_dir, str(i + 1))
        process_image_recursive(sub_image, sub_image_dir, ocr, predictor, level + 1, max_level, messages, image_id + 1, box_counts)

def find_deepest_directories(base_dir):
    deepest_dirs = []
    for root, dirs, _ in os.walk(base_dir):
        if not dirs:
            deepest_dirs.append(root)
    return deepest_dirs

def parse_args():
    parser = argparse.ArgumentParser(description="Process images using Detectron2 and OCR")
    parser.add_argument('--base_image_dir', type=str, required=True, help="Base directory for input images")
    parser.add_argument('--base_output_dir', type=str, required=True, help="Base directory for output images")
    parser.add_argument('--use_cuda', action='store_true', help="Flag to enable CUDA (default is to use CPU if not set)")
    parser.add_argument('--message_file', type=str, required=True, help="Path to save the messages.json file")
    parser.add_argument('--confidence_threshold', type=float, default=0.35, help="Confidence threshold for detections (default is 0.35)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # CUDA_VISIBLE_DEVICES must be a GPU index string ('0') or '' to hide all GPUs;
    # 'cpu' is not a valid value and silently ignored by CUDA runtime.
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' if args.use_cuda else ''
    
    predictor = setup_predictor(use_cuda=args.use_cuda, confidence_threshold=args.confidence_threshold)
    ocr = setup_ocr()
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    messages = []
    box_counts_per_image = {}

    # Get all image paths for processing (for tqdm total)
    image_tasks = []
    deepest_dirs = find_deepest_directories(args.base_image_dir)
    for dir_path in deepest_dirs:
        for filename in os.listdir(dir_path):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_path = os.path.join(dir_path, filename)
                file_name = os.path.splitext(filename)[0]
                specific_output_dir = os.path.join(args.base_output_dir, os.path.basename(dir_path), file_name)
                box_count_path = os.path.join(specific_output_dir, "box_counts.json")
                if not os.path.exists(box_count_path):
                    image_tasks.append((image_path, specific_output_dir, file_name))

    # Add progress bar
    from tqdm import tqdm
    pbar = tqdm(total=len(image_tasks), desc="Processing Images")

    for image_path, specific_output_dir, file_name in image_tasks:
        # Skip already processed images (checkpoint)
        box_count_path = os.path.join(specific_output_dir, "box_counts.json")
        if os.path.exists(box_count_path):
            pbar.update(1)
            continue

        image = load_image(image_path)
        if image is None:
            pbar.update(1)
            continue

        os.makedirs(specific_output_dir, exist_ok=True)
        save_image(image, os.path.join(specific_output_dir, "original.jpg"))

        box_counts = {}
        process_image_recursive(image, specific_output_dir, ocr, predictor, messages=messages, box_counts=box_counts)
        box_counts_per_image[file_name] = box_counts

        with open(box_count_path, "w") as f:
            json.dump(box_counts, f, indent=4)

        pbar.update(1)

    pbar.close()

    with open(args.message_file, "w") as f:
        json.dump(messages, f, indent=4)

    with open(os.path.join(args.base_output_dir, "box_counts.json"), "w") as f:
        json.dump(box_counts_per_image, f, indent=4)
