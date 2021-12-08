import streamlit as st
import cv2
import FaceMaskDet
import FriendDet
# from streamlit_webrtc import webrtc_streamer #Figure out how to use this

# webrtc_streamer(key="example")
## Sidebar
st.sidebar.title("Face unmasked")
st.sidebar.info("This is a demo of a Streamlit WebRTC app")
# st.sidebar.selectbox("Select a video source", ["Mask Detection", "Friend detection"])
selection = st.sidebar.selectbox("Select a video source", ["Mask Detection", "Friend Detection"])

## Main webcam
st.title(selection)
if selection == "Mask Detection":
    FaceMaskDet #Initialize the mask detection
elif selection == "Friend Detection":
   FriendDet # Initialize the friend detection
