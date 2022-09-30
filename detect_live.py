import torch
import numpy as np
import easyocr
import cv2
from detect_image import detect_plate
from settings import *

def detect_from_cam(image_size: tuple[int], model, ocr_model=None, readtext=True):
    if ocr_model is None:
        ocr_model = easyocr.Reader(['en'])
    
    if isinstance(model, str):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model, _verbose=False)
        
    cam = cv2.VideoCapture(0)
    h, w = cam.get(cv2.CAP_PROP_FRAME_HEIGHT), cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    crop = int((w-h)/2)
    
    while True:
        ret, frame = cam.read()
        if ret:
            frame = frame[:, crop: -crop, ...]
            frame = cv2.resize(frame, image_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if readtext:
                ren_frame, _, texts = detect_plate(frame, model, image_size, ocr_model, False)
                for t in texts:
                    if t: print(t)
                    
            else:
                out = model(frame)
                ren_frame = np.squeeze(out.render())
            
            ren_frame = cv2.cvtColor(ren_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Detecting License Plates...", ren_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    detect_from_cam(IMAGE_SIZE, MODEL_PATH, readtext=False)