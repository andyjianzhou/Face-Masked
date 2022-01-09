import streamlit as st
import cv2
import streamlit as st
from multiapp import MultiApp
from apps import FaceMaskDet, UploadDet # import your app modules here
from faster_utils import FaceMaskDataset, get_predictions, get_model_instance_segmentation
#streamlit
import streamlit as st
# from navbar import navbar

def navigation():
        try:
            path = st.experimental_get_query_params()['p'][0]
        except Exception as e:
            # st.error('Please use the main app.')
            return None
        return path
page_bg_img = '''
<style>
.stApp {
background-image: url("https://www.wallpaperflare.com/static/198/774/447/red-background-green-red-blue-wallpaper.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

button_style = '''
<style>
.css-1bvhuai {
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 9.75rem;
    border-radius: 16.25rem;
    border-radius: -0.75rem; 0px: ;
    line-height: 1.6;
    color: inherit;
    width: 60%;
    user-select: none;
    background-color: rgb(43, 44, 54);
    border: 1px
    solid rgba(250, 250, 250, 0.2);
}
.css-1bvhuai:focus:not(:active) {
    border-color: rgb(247 247 247);
    color: rgb(255 255 255);
}
.css-1bvhuai:hover {
    border-color: rgb(247 247 247);
    color: rgb(247 247 247);
}

.css-1bvhuai:focus {
    box-shadow: rgb(230 230 250 / 50%) 0px 0px 0px 0.2rem;
    outline: none;
}

.st-cs {
    background-color: rgb(119 75 255);
}
.st-ei {
    background-color: rgb(164 72 233);
}
</style>
'''
st.markdown(button_style, unsafe_allow_html=True)

text_style = '''
<style>
.css-ng1t4o {
    background-color: rgb(251 251 253);
    background-attachment: fixed;
    flex-shrink: 0;
    height: 100vh;
    overflow: auto;
    padding: 6rem 1rem;
    position: relative;
    transition: margin-left 300ms ease 0s, box-shadow 300ms ease 0s;
    width: 21rem;
    z-index: 100;
    margin-left: 0px;
}
h1 {
    font-family: "Source Sans Pro", sans-serif;
    font-weight: 700;
    color: rgb(56 55 55);
    padding: 1.25rem 0px 1rem;
    margin: 0px;
    line-height: 1.4;
}
.st-cj {
    color: rgb(0 0 0);
}
</style>
'''
st.markdown(text_style, unsafe_allow_html=True)

sidebar_style = '''
<!--    start of exit button sidebar-->
<style>
.css-d1b1ld {
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    border-radius: 0.25rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: auto;
    user-select: none;
    background-color: #872dd9cc;;
    border: 1px solid transparent;
    padding: 0.75rem;
}
<!--    hover exit button-->
.css-d1b1ld:focus:not(:active) {
    border-color: rgb(0 0 0);
    color: rgb(0 0 0);
}
.css-d1b1ld:hover {
    border-color: rgb(0 0 0);
    color: rgb(0 0 0);
}
.css-d1b1ld:focus {
    box-shadow: #c485fd 0px 0px 0px 0.2rem;
    outline: none;
}   
</style>
'''
st.markdown(sidebar_style, unsafe_allow_html=True)


def main():


    ## Sidebar
    st.sidebar.title("Face Masked") 
    if st.sidebar.button("About", key="about"):
        st.markdown(''' 
        <div>
        <style>
            .content h1{
                animation-duration: 8s;
                animation-name: slidein;
                animation-iteration-count: infinite;
            }
            @keyframes slidein {
                0% {
                    margin-left: 0%;
                }
                50% {
                    margin-left: 150px;
                }
                100% {
                    margin-left: 0%;
                }
            }
            .content p{
                font-size: 20px;
                font-family: "Source Sans Pro", sans-serif;
                color: rgb(56 55 55);
            }
            .content {
                margin-left: -50%;
                margin-top: 10%;
            }
            .content .btn-unique{
                display: inline-block;
                background: linear-gradient(45deg, #87adfe, #b452fb);
                border-radius: 6px;
                paddding: 10px 20px;
                box-sizing: border-box;
                text-decoration: none;
                color: #fff;
                box-shadow: 3px 8px 22px rgba(94. 28, 68, 0.15);
            s}
        </style>
         ''', unsafe_allow_html=True)

        about_content = '''
        <div class="content">
            <a href="" class="btn-unique">Using State-of-the-Art Object detection methods!</a>
            <h1>Ever wondered if people <br> wore their <br> mask?</h1>
            <p>An app that detects the presence of a face mask.</p>
        </div>
        '''
        st.markdown(about_content, unsafe_allow_html=True)

        
    if st.sidebar.button("Contact", key="contactt"):
        contact_content = '''
        <div class="contact-content"> 
            <h1>Contact</h1>
        </div>'''
        st.markdown(contact_content, unsafe_allow_html=True)
    
    if st.sidebar.button("Help", key="help"):
        st.write("Help")

    if navigation() == "usage":
        st.markdown('''
        <div class="usage">
            <h1>How it works</h1>
            <p>On the sidebar, you can see the different options.</p>
            ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

    # if navigation() is None:
    #     st.markdown('''
    #     <style>
    #     .content p{
    #         font-size: 20px;
    #         font-family: "Source Sans Pro", sans-serif;
    #         color: rgb(56 55 55);
    #     }
    #     .content {
    #         margin-left: -50%;
    #         margin-top: 10%;
    #     }
    #     .content .btn-unique{
    #         display: inline-block;
    #         background: linear-gradient(45deg, #87adfe, #b452fb);
    #         border-radius: 6px;
    #         paddding: 10px 20px;
    #         box-sizing: border-box;
    #         text-decoration: none;
    #         color: #fff;
    #         box-shadow: 3px 8px 22px rgba(94. 28, 68, 0.15);
    #     }
    #     </style>
    #      ''', unsafe_allow_html=True)

    #     about_content = '''
    #     <div class="content">
    #         <a href="/?p=usage" class="btn-unique">See how it works!</a>
    #         <h1>Ever wondered if people <br> wore their <br> mask?</h1>
    #         <p>An app that detects the presence of a face mask.</p>
    #     </div>
    #     '''
    #     st.markdown(about_content, unsafe_allow_html=True)
        
    app = MultiApp()
    app.add_app("Face Mask Detection", FaceMaskDet.app)

    # app.add_app("Friend Detection", FriendDet.app)
    # specify the primary menu definition
    #get the id of the menu item clicked
    # if(navigation() == 'About'):
    #     # use htmle
    #     # st.markdown("""""")
    #     st.write("This is About Page")  
    # if(navigation() == 'YouTube'):  
    #     st.write("Youtube Page")    
    # if(navigation() == 'Demos'):
    #     st.write("Demos Page")
    # The main app
    app.run()   
