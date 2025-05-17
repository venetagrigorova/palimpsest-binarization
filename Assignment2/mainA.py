from ultralytics import YOLO
import os
import cv2

def main():
    current_dir = os.path.dirname(__file__)
    image_dir = os.path.join(current_dir,"data", "images")
    model_dir = os.path.join(current_dir,"models")
    
    model = YOLO(os.path.join(model_dir,"yolov8n.pt"))
    
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        layout_results = model(image)[0]
        
        for _, box in enumerate(layout_results.boxes):
            cls = int(box.cls[0])
            # only interested in printed text regions
            if cls == 2:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                crop = image[y1:y2, x1:x2]
                cv2.show("Cropped Image", crop) 
        break
        
    
    

if __name__ == "__main__":
    main()