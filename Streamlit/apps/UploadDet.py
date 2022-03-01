import streamlit as st
from PIL import Image
#Opencv and torch utils
import cv2
import torch
from faster_utils import FaceMaskDataset, get_predictions, get_model_instance_segmentation, torch_to_pil, plot_img_bbox, apply_nms
from win10toast import ToastNotifier
toast = ToastNotifier()
# from utils import FaceMaskDataset, get_transform, get_model, get_device, get_dataloader
PATH = 'UploaderDet'

def load_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return image, width, height
def app():
    st.title("Upload an Image") 
    image = st.file_uploader("Upload an image...", type=["jpg"], key="Uploaderdet")
    if image:
        if image is not None:
            # file_details = {"FileName":image.name,"FileType":image.type,"FileSize":image.size}
            # st.write(file_details)
            #Loading in image
            img, width, height = load_image(image)
            #Using function from faster_utils to get predictions
            img, preds = get_predictions(img, width=width, height=height, PATH=PATH, real_time = False) 
            print('predicted #boxes: ', preds['labels']) #debugging purposes
            print('predicted #boxes: ', len(preds['boxes']))
            #apply nms(Non Max Suppression) to get rid of overlapping boxes
            nms_preds = apply_nms(preds, 0.7)
            # print('real #boxes: ', len(target['labels']))
            image_display, label = plot_img_bbox(torch_to_pil(img), nms_preds, PATH)
            st.image(image_display)

            if label == 'Without Mask':
                toast.show_toast("Face Masked Alert","Please wear your mask!",duration=5,icon_path="Face-Mask.ico")
            elif  label == 'Mask Weared Incorrect':
                toast.show_toast("Face Masked Alert","Wear your mask properly!",duration=5,icon_path="Face-Mask.ico")
