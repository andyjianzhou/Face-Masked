import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#Opencv and torch utils
import cv2
import torch
from faster_utils import FaceMaskDataset, get_predictions, get_model_instance_segmentation, torch_to_pil, plot_img_bbox, apply_nms
from win10toast import ToastNotifier
from zipfile import ZipFile
import os

from random import randint

if 'key' not in st.session_state:
    st.session_state.key = str(randint(1000, 100000000))

toast = ToastNotifier()
# from utils import FaceMaskDataset, get_transform, get_model, get_device, get_dataloader
# Sets model path
PATH = 'FriendDet'

#Save files function
def save_uploadedfile(uploadedfile):
    with open(os.path.join("fileDir",uploadedfile.name),"wb") as f:
        f.write((uploadedfile).getbuffer())
    return st.success("Saved File:{} to fileDir".format(uploadedfile.name))

#Loading image function using PIL 
def load_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return image, width, height
def app():
    #Bulk select option/decorations
    st.markdown('''
    <style>
        .st-cq {
        border-bottom-color: rgb(186 75 255);
        }
        .st-cp {
        border-top-color: rgb(219 75 255);
        }
        .st-co {
        border-right-color: rgb(194 75 255);
        }
        .st-cn {
        border-left-color: rgb(202 75 255);
        }
        .st-ei {
        background-color: rgb(172 99 245);
        
        }   
        .css-e3kofv:hover, .css-e3kofv:active, .css-e3kofv:focus {
        background: rgb(201 163 235);
        }
    </style>
    ''', unsafe_allow_html=True)
    zipObj = ZipFile('test.zip', 'w')

    st.title("Upload an Image") 
    #Using streamlit inbuilt function to Bulk select option and upload image
    options = st.multiselect("Choose which person to download", ["Jun", "Nic"])
    if not options:
        st.error("Please select a person to download")
    else:
        st.write(f"You selected option {options}")
        #File uploader provided by streamlit
        image = st.file_uploader("Upload an image...", type=["jpg", "png"], key="frienddet", accept_multiple_files=True)
        print(image)
        if image:
            if image is not None:
                #create a file so that we can save the image, later retrieve with zip file
                try:
                    os.mkdir("fileDir")
                except FileExistsError:
                    print('Directory not created')
                for i in image:
                    #load image using PIL
                    img, width, height = load_image(i)
                    #get predictions from faster_utils      
                    img, preds = get_predictions(img, width=width, height=height, PATH=PATH, real_time = False) 
                    print('predicted #boxes: ', preds['labels']) #debugging purposes
                    print('predicted #boxes: ', len(preds['boxes']))
                    
                    print(options)
                    #apply nms(Non Max Suppression) to get rid of overlapping boxes
                    nms_preds = apply_nms(preds, 0.7)
                    image_display, label = plot_img_bbox(torch_to_pil(img), nms_preds, PATH)
                    print(label)
                    if options == ["Jun"]:
                        
                        if label == "Jun Mask On" or label == "Jun Mask Off":
                            st.image(image_display, 50)
                            save_uploadedfile(i)
                            #if equals to name, write into zip file
                            zipObj.write(f'fileDir/{i.name}')
                            print("Saved!")
                        else:
                            personFound = False
                            st.error("Person not found")

                    elif options == ["Nic"]:
                        if label == "Nic Mask On" or label == "Nic Mask Off":
                            st.image(image_display, width=50)
                            
                            save_uploadedfile(i)
                            #if equals to name
                            zipObj.write(f'fileDir/{i.name}')
                            print("Saved!")
                        else:
                            personFound = False
                            st.error("Person not found")

                    elif options == ["Nic", "Jun"] or ["Jun, Nic"]:
                        if label == "Nic Mask On" or label == "Nic Mask Off" or label == "Jun Mask On" or label == "Jun Mask Off":
                            st.image(image_display, width=50)
                            save_uploadedfile(i)
                            #if equals to name
                            zipObj.write(f'fileDir/{i.name}')
                            print("Saved!")
                        

                
            zipObj.close()
            with open("test.zip", "rb") as fp:
                if( not zipObj.namelist() == []):
                    if st.download_button("Download images of friends!", data=fp, file_name="test.zip", mime="application/zip"):
                        st.session_state.key = str(randint(1000, 100000000))
                        st.sync()