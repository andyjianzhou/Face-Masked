import streamlit as st

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