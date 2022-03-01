import streamlit as st
import cv2
from faster_utils import *
import time
import torch
from PIL import Image
from io import BytesIO
from apps import UploadDet

import os.path
from os import path
## Use yolov5 model inspired from github https://github.com/taeokimeng/object-detection-yolo/blob/main/detection/object_detection.py 
## Increase fps, lowers performance but that's not the motive, higher fps


def load_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return image, width, height
# Inputs into app folder, will create a better webapp
def app():

    st.markdown('''
    <style>
        .rectangle-bg{
            height: 66.67vw;
        }
    ''', unsafe_allow_html=True)
    rectangle_bg = '''
        <div class = "rectangle-bg" style="background-color: #FFFFFF; padding: 10px;"> 
        </div>  
    ''' 
    # st.markdown(rectangle_bg, unsafe_allow_html=True)
    
    st.title("Webcam Live Feed")
    run = st.checkbox('Detect')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0) #enable webcam
    #fps usage, calculate e fps
    new_frame_rate = 0
    prev = 0
    #yolov5 prediction test
    
    # real time yolov5
    #load the model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False, classes=3, device = 'cpu')
    #obtain the yolov5 weights
    model = torch.hub.load('apps\yolov5', 'custom', path = '../yolov5/runs/train/exp3/weights/best.pt', source = 'local')
    print("Completed loading yolov5 model")
    #set model to eval mode
    model.eval()
    
    t = st.empty()
    while run:
        print("Webcam is running")
        ret, frame = cap.read()
        if ret:
            new_frame_rate = time.time()
            fps = 1/(new_frame_rate-prev)
            prev = new_frame_rate   
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = model(frame)
            output.render()
    
            for img in output.imgs:
                buffered = BytesIO()
                img_base64 = Image.fromarray(img)
                img_base64.save(buffered, format="JPEG")
        
            print(fps) #prints fps off
            FRAME_WINDOW.image(img_base64)
            
            array = output.pandas().xyxy[0]['class'].values
            print(array)
            if 2 in array:
                t.text("Wear your mask!")
            elif 0 in array:
                t.text("Mask weared incorrectly!")
            else:
                t.text("You're a good human being")

            # if output.pandas().xyxy[0]['name'] == 'without_mask':
                
        else:
            cap.release() #Stop the camera
            run = False 
    else:
        st.write('Stopped')
    
    UploadDet.app()
    
    #Test usages
    # image = st.file_uploader("Upload an image...", type=["jpg"], key="facedet")
    # if image:
    #     if image is not None:
    #         file_details = {"FileName":image.name,"FileType":image.type,"FileSize":image.size}
    #         st.write(file_details)

          

