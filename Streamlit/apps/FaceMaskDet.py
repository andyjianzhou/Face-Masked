import streamlit as st
import cv2
from utils import *
import time
## Use yolov4 model inspired from github https://github.com/taeokimeng/object-detection-yolo/blob/main/detection/object_detection.py 
## Increase fps, lowers performance but that's not the motive, higher fps
def load_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return image, width, height
# Inputs into app folder, will create a better webapp
def app():
    
    st.title("Webcam Live Feed")
    start_frame_number = 30 #Trying to skip frames so model won't detect on every frame, unsuccessful 
    run = st.checkbox('Detect')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0) #enable webcam
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number) #set amount of frames to skip
    #fps usage, calculate fps
    new_frame_rate = 0
    prev = 0
    while run:
        print("Webcam is running")
        ret, frame = cap.read()
        if ret:
            new_frame_rate = time.time()
            fps = 1/(new_frame_rate-prev)
            prev = new_frame_rate
            
            #copy image frame to get dimensions
            image_dim = frame.copy()
            image_dim = cv2.cvtColor(image_dim, cv2.COLOR_BGR2RGB)
            width = 700
            height = image_dim.shape[1]
            # frame = load_image(frame)
            frame, preds = get_predictions(frame, width, height, real_time = True) #imported from utils.py
            
                
            #apply nms
            nms_preds = apply_nms(preds, 0.7) #Advanced nms application

            frame = plot_img_bbox(torch_to_pil(frame), nms_preds) #display frame
            print(fps) #prints fps off
            FRAME_WINDOW.image(frame)
        else:
            cap.release() #Stop the camera
            run = False 
    else:
        st.write('Stopped')
          

