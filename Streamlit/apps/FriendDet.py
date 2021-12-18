import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#Opencv and torch utils
import cv2
import torch
from utils import FaceMaskDataset, get_predictions, get_model_instance_segmentation, torch_to_pil, plot_img_bbox, apply_nms
# from utils import FaceMaskDataset, get_transform, get_model, get_device, get_dataloader
def load_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return image, width, height
def app():
    st.title("Friend Detection") 
    image = st.file_uploader("Upload an image...", type=["jpg"])
    if image:
        if image is not None:
            file_details = {"FileName":image.name,"FileType":image.type,"FileSize":image.size}
            st.write(file_details)

        img, width, height = load_image(image)
        img, preds = get_predictions(img, width=width, height=height, real_time = False) 
        print('predicted #boxes: ', len(preds['labels'])) #debugging purposes
        print('predicted #boxes: ', len(preds['boxes']))
        #apply nms
        nms_preds = apply_nms(preds, 0.7)
        # print('real #boxes: ', len(target['labels']))
        image_display = plot_img_bbox(torch_to_pil(img), nms_preds)

        
        st.image(image_display)
        # FRAME_WINDOW = st.image([])
        # FRAME_WINDOW.image(image)

