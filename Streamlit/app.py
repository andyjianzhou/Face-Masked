import streamlit as st
import cv2
import streamlit as st
from multiapp import MultiApp
from apps import FriendDet, FaceMaskDet # import your app modules here
import torch
from utils import FaceMaskDataset, get_predictions, get_model_instance_segmentation
#streamlit
import streamlit as st
from navbar import navbar

def navigation():
    try:
        path = st.experimental_get_query_params()['p'][0]
    except Exception as e:
        st.error('Please use the main app.')
        return None
    return path
navbar()
app = MultiApp()

## Sidebar
st.sidebar.title("Face Masked")

app.add_app("Face Mask Detection", FaceMaskDet.app)
app.add_app("Friend Detection", FriendDet.app)
# specify the primary menu definition
#get the id of the menu item clicked
if(navigation() == 'About'):
    st.write("This is About Page")
if(navigation() == 'YouTube'):
    st.write("Youtube Page")
if(navigation() == 'Demos'):
    st.write("Demos Page")
# The main app
app.run()