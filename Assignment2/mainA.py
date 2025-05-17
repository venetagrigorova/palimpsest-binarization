from ultralytics import YOLO
import os
import cv2

def main():
    current_dir = os.path.dirname(__file__)
    image_dir = os.path.join(current_dir,"data", "images")
    model_dir = os.path.join(current_dir,"models")
    output_dir = os.path.join(current_dir,"data", "text_images")
    
    model = YOLO(os.path.join(model_dir,"yolov8n.pt"))
    
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        layout_results = model(image)[0]
        prefix = image_name.split("_")[0] + "_ImageFile"
        text_image_count = 0
        for _, box in enumerate(layout_results.boxes):
            cls = int(box.cls[0])
            # only interested in printed text regions
            if cls == 2:
                text_image_count += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                text_image = image[y1:y2, x1:x2]
                text_image_name = f"{prefix}_text_{text_image_count}.jpg"
                text_image_path = os.path.join(output_dir,text_image_name)
                cv2.imwrite(text_image_path, text_image)
        
    
    

if __name__ == "__main__":
    main()