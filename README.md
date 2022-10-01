# License Plate Detection 🚗🚔

This License plate detector can detect and read license plates, from images as well as live using camera. The object detection is done using YOLOv5 and Optical Character Recognition using 'EasyOCR' python library.

YOLO (You Only Look Once, Joseph Redmon et al.) is an exteremly fast and lightweight object detection algorithm highly preferred for live detection systems. [YOLOv5](https://github.com/ultralytics/yolov5/) is the PyTorch implementation of the yolo algorithm by Ultralytics.

See the example images provided in the dropdown below. These detections are done using this program.

<details>
<summary> Show Examples </summary>
<img src="https://raw.githubusercontent.com/ZenithFlux/assets/main/license_plate_detector/detect1.jpg" title="Single Car">
</br></br>
<img src="https://raw.githubusercontent.com/ZenithFlux/assets/main/license_plate_detector/detect2.jpg" title="Cars in traffic">
</br></br>
<img src="https://raw.githubusercontent.com/ZenithFlux/assets/main/license_plate_detector/detect3.jpg" title="Car park">
</br></br>
<img src="https://raw.githubusercontent.com/ZenithFlux/assets/main/license_plate_detector/detect4.jpg" title="Image from Dashcam">
</br></br>
<img src="https://raw.githubusercontent.com/ZenithFlux/assets/main/license_plate_detector/detect5.jpg" title="Car in speed/Blurry Image">
</br></br>
<img src="https://raw.githubusercontent.com/ZenithFlux/assets/main/license_plate_detector/detect6.jpg" title="Cartoon Car">
</details>
</br>

In this project, yolo algorithm is combined with EasyOCR to read license numbers on plates in real time.

## Steps to use the model🚀

**Note:-** Instructions on how to train your own model are provided in 'build_model.ipynb'. Folowing steps are specifically on how to use a trained model for plate detection.

Steps -

1. [Install PyTorch](https://pytorch.org/get-started/locally/) according to your pc specs.

2. To install other dependencies, enter the following commands in command line sequentially-  
```console
pip install -r requirements.txt
pip uninstall -y opencv-python-headless
pip install --force-reinstall opencv-python
```
Ignore Errors produced by pip about *missing opencv-python-headless*.<br/><br/>

3. Provide path to a YOLOv5 model file in *settings.py* file as MODEL_PATH.  
Also set the IMAGE_SIZE on which the model was trained.  
[Download](https://drive.google.com/file/d/1fZIv3KQ8nBUe6YhnQ2wtLd0TBgsLOifE/view?usp=sharing) model trained by me on 640x640 images or use any other YOLOv5 model.

4. Run file *'detect_image.py'* to detect text from license plates in images.  
Run file *'detect_live.py'* to detect license plates live from camera.

## Dataset used for training 🚂

[Kaggle Car License Plate Detection dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)