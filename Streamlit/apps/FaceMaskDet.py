import streamlit as st
import cv2
from utils import *
import time

def load_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return image, width, height

def app():
    st.title("Webcam Live Feed")
    start_frame_number = 30
    run = st.checkbox('Detect')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
    #fps usage
    new_frame_rate = 0
    prev = 0
    while run:
        print("Webcam is running")
        ret, frame = cap.read()
        if ret:
            new_frame_rate = time.time()
            fps = 1/(new_frame_rate-prev)
            prev = new_frame_rate

            image_dim = frame.copy()
            image_dim = cv2.cvtColor(image_dim, cv2.COLOR_BGR2RGB)
            width = 700
            height = image_dim.shape[1]
            # frame = load_image(frame)
            frame, preds = get_predictions(frame, width, height, real_time = True)
            
                
            #apply nms
            nms_preds = apply_nms(preds, 0.7)

            frame = plot_img_bbox(torch_to_pil(frame), nms_preds)
            print(fps)
            FRAME_WINDOW.image(frame)
        else:
            cap.release()
    else:
        st.write('Stopped')

