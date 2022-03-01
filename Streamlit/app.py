import streamlit as st
import streamlit as st
from multiapp import MultiApp
from apps import FaceMaskDet, FriendDet # import your app modules here
from faster_utils import FaceMaskDataset, get_predictions, get_model_instance_segmentation
#streamlit
import streamlit as st
# from navbar import navbar

#This is to find url code
import streamlit.components.v1 as components
def navigation():
        try:
            path = st.experimental_get_query_params()['p'][0]
        except Exception as e:
            # st.error('Please use the main app.')
            return None
        return path

# This is to decorate streamlit page using custom html and css
# print("File exists:" + str(path.exists("intro.htm")))
HtmlFile = open("intro.htm", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
print(source_code)
components.html(source_code)
page_bg_img = '''
<style>
    .stApp {
        background-image: url("https://cdn.dribbble.com/users/2272349/screenshots/7207200/ocean-guardian-pulse-loop-by-the-sound-of-breaking-glass.gif");
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
        color: #fff;
        cursor: pointer;
    }
    <!-- Worry about button deco later -->
    .css-1bvhuai:before {
        transtion: 0.5s all ease;
        position: relative;
        top: 0;
        left: 50%;
        right: 50%;
        bottom:0;
        opacity: 0;
        contennt:"";
        background-color: rgb(176, 43, 196);
    }
    .css-1bvhuai:hover:before {
        transtion: 0.5s all ease;
        left:0;
        right:0;
        opactiy: 1;
        z-index: -1;
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
    color: rgb(255 255 255);
    padding: 1.25rem 0px 1rem;
    margin: 0px;
    line-height: 1.4;
}
.st-cj {
    color: rgb(155 155 155);
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
    st.sidebar.image("https://pngimg.com/uploads/medical_mask/medical_mask_PNG48.png", use_column_width=True)
    if st.sidebar.button("About", key="about"):
        st.markdown(''' 
        <style>
            h2{
                animation-duration: 8s;
                animation-name: slidein;
                animation-iteration-count: infinite;
            }
            @keyframes slidein {
                0% {
                    margin-left: 0%;
                }
                50% {
                    margin-left: 50px;
                }
                100% {
                    margin-left: 0%;
                }
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
            }
            p{
                margin-left:-50%;
            }
            
        </style>
         ''', unsafe_allow_html=True)

        st.markdown('''
        <style>
            .about-image {
                margin-left:50%;
                margin-top:-50%;
            }
            .about-image {
                animation: fadeIn 5s;
                -webkit-animation: fadeIn 5s;
                -moz-animation: fadeIn 5s;
                -o-animation: fadeIn 5s;
                -ms-animation: fadeIn 5s;
                }
                @keyframes fadeIn {
                0% {opacity:0;}
                100% {opacity:1;}
                }

                @-moz-keyframes fadeIn {
                0% {opacity:0;}
                100% {opacity:1;}
                }

                @-webkit-keyframes fadeIn {
                0% {opacity:0;}
                100% {opacity:1;}
                }

                @-o-keyframes fadeIn {
                0% {opacity:0;}
                100% {opacity:1;}
                }

                @-ms-keyframes fadeIn {
                0% {opacity:0;}
                100% {opacity:1;}
                }
         ''', unsafe_allow_html=True)

        about_content = '''
        <div class="content">
            <a href="" class="btn-unique">Using State-of-the-Art Object detection methods!</a>
            <h2>Ever wondered if people <br> wore their <br> mask?</h1>
        </div>

        <div class="details">
            <p>An app that detects the presence of a face mask.</p>
            <img src="https://cdn.discordapp.com/attachments/741006860957778031/929858542188519554/about_image.png", width=750, height=750 class="about-image">
        </div>
        '''
        st.markdown(about_content, unsafe_allow_html=True)
    
    if st.sidebar.button("Help", key="help"):
        st.title("How to use the website:")

        st.header("Nav Bar:")
        st.write("Here, you can navigate through the website using this interactive navbar.")
        st.image(
            "https://cdn.discordapp.com/attachments/700087629323960351/947640390654263376/unknown.png",
            width=600, # Manually Adjust the width of the image as per requirement
        )
        st.header("Face Mask detection checkbox:")
        st.image(
            "https://cdn.discordapp.com/attachments/700087629323960351/947638906201972827/unknown.png",
            width=600, # Manually Adjust the width of the image as per requirement
        )
        st.write("Here, you can use the app to detect the presence of a face mask. Pressing the detect button will send a live feed like so:")
        st.image(
            "https://cdn.discordapp.com/attachments/700087629323960351/947639982682685490/unknown.png",
            width=600, # Manually Adjust the width of the image as per requirement
        )
        st.write("If you see a face with a mask, the app will detect it and show a box around it. Depending on whether you are wearing a mask or not, the box colour will change, so will the labels.")
        st.write("If you are wearing a mask, the box will be pink. If you are not wearing a mask, the box will be orange. If you are wearing your mask incorrectly, the box will be red.")
        st.write("It will also alert the user if they are wearing a mask or not.")

        st.subheader("Face Mask Uploader:")
        st.write("This uses a more accurate model, the FasterRCNN RPN model, to detect the presence of a face mask. The model is trained on the COCO dataset, which is a large dataset of images of people and their masks. It uses the Region Proposal network which is a CNN algorithm, that's widely used in the RCNN family of models for object detection.")
        st.image(
            "https://cdn.discordapp.com/attachments/700087629323960351/947652044980891699/unknown.png",
            width=600, # Manually Adjust the width of the image as per requirement
        )
        st.write("Look! It has 99% accuracy on ANY photo!")
        st.image(
            "https://cdn.discordapp.com/attachments/700087629323960351/947655450797686834/unknown.png",
            width=800, # Manually Adjust the width of the image as per requirement
        )
        st.write("I used advanced algorithms in my model in order to accomplish this feat, mosiac augmentation, windowing on the image, a custom loss function using adam optimizer, and a a new SOTA metric(weighted area under the curve) for bounding box accuracy.")
        st.write("Realize how there is also a notification alert to alert the user that someone isn't wearing their mask!")

        st.subheader("Friend detection checkbox:")

        st.write("You can decide to add a friend's face on the app and upload images to train the model. The model will detect your desired person that you would like to keep, and download them in a neat temprorary file.")
    
        st.image(
            "https://cdn.discordapp.com/attachments/700087629323960351/947656334352998420/unknown.png",
            width=600, # Manually Adjust the width of the image as per requirement
        )
    if navigation() == "usage":
        st.markdown('''
        <div class="usage">
            <h1>How it works</h1>
            <p>On the sidebar, you can see the different options.</p>
            ''', unsafe_allow_html=True)
    

if __name__ == "__main__":
    #Run new html page
    main()
    # Run the applets
    app = MultiApp()
    #Add app pages onto "MultiApp()" => Creates session states
    app.add_app("Face Mask Detection", FaceMaskDet.app)
    app.add_app("Friend Detection", FriendDet.app)  
    
    app.run()   
