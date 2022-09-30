import os
import torch
import cv2
import numpy as np
import easyocr
from settings import *

def detect_plate(img, model, img_size: tuple[int], ocr_model=None, is_BGR=True):
    if ocr_model is None:
        ocr_model = easyocr.Reader(['en'])
    
    if isinstance(model, str):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model, _verbose=False)
    
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.resize(img, img_size)
        
    elif isinstance(img, np.ndarray):
        img = cv2.resize(img, img_size)
    
    if is_BGR:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    output = model(img.copy())
    out = output.xyxy[0].cpu().numpy()
    ren_image = np.squeeze(output.render())
    
    text = []
    plate_images = []
    for plate in out:
        xmin, ymin, xmax, ymax = plate.astype('int32').tolist()[0:4]
        plate_img = img[ymin:ymax, xmin:xmax, ...]
        txt = ocr_model.readtext(plate_img, detail=0)
        text.append(txt)
        plate_images.append(plate_img)
    
    return ren_image, plate_images, text
    
    
if __name__ == "__main__":
    path = input("Enter image path: ")
    save_folder = input("\nEnter save location for images of detected plates (Leave blank if you don't want to save):\n")
    
    _, imgs, texts = detect_plate(path, MODEL_PATH, IMAGE_SIZE)
    
    print("\nLicense plate texts detected in image-")
    for i in range(len(texts)):
        print(texts[i])
        if save_folder:
            out = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_folder, f"{i}.jpg"), out)