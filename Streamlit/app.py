import streamlit as st
from streamlit_webrtc import webrtc_streamer

webrtc_streamer(key="example")
st.write("Hello World")