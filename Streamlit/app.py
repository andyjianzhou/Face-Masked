import streamlit as st
import cv2
import streamlit as st
from multiapp import MultiApp
from apps import FriendDet, FaceMaskDet # import your app modules here
import torch
from utils import FaceMaskDataset, get_predictions, get_model_instance_segmentation

app = MultiApp()
# from streamlit_webrtc import webrtc_streamer #Figure out how to use this

# webrtc_streamer(key="example")
## Sidebar
st.sidebar.title("Face Masked")
# specify the primary menu definition

    
#get the id of the menu item clicked
app.add_app("Face Mask Detection", FaceMaskDet.app)
app.add_app("Friend Detection", FriendDet.app)
# The main app
app.run()
