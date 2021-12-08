import streamlit as st
import cv2
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
    st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    while run:
        print("Webcam is running")
        ret, frame = camera.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
        else:
            print("Error")
            break
    else:
        st.write('Stopped')

    print("Webcam released")
elif selection == "Friend Detection":
  st.title("Friend Detection") 
  st.title("Webcam Live Feed") 