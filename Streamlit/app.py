import streamlit as st
import cv2
import streamlit as st
from multiapp import MultiApp
from apps import FriendDet, FaceMaskDet # import your app modules here

app = MultiApp()
# from streamlit_webrtc import webrtc_streamer #Figure out how to use this

# webrtc_streamer(key="example")
## Sidebar
st.sidebar.title("Face Masked")

app.add_app("FaceMaskDet", FaceMaskDet.app)
app.add_app("FriendDet", FriendDet.app)
# The main app
app.run()
