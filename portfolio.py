import cv2
import os
import streamlit as st  
import toml
import google.generativeai as genia
import numpy as np
from mediaFunc import *
from segmenter import *
import mediapipe as mp

st.set_page_config(
    page_title="Juan's Portfolio",
    page_icon="./iniciales.ico",
    #layout="wide",
    initial_sidebar_state="expanded",
)
##

##############################
# configuration
api_key=""
with open('./keys.toml','r') as f:
    config = toml.load(f)
    api_key=config['keys']['google']
genia_key_api = os.environ.get('GENIA_API_KEY', api_key)
print("llave",genia_key_api)
genia.configure(api_key=genia_key_api)
about_short= config["about"]['ing']
model = genia.GenerativeModel('gemini-1.5-flash')
main_container=st.container(height=None,border=True)
########################


########################
#   clean input text value
def reset_input(llave:str):
    saved_key='saved-'+llave
    if llave not in st.session_state:
        st.session_state[llave]=""
    if saved_key not in st.session_state:
        st.session_state[saved_key]=""
    st.session_state[saved_key]=st.session_state[llave]
    st.session_state[llave]=""
################    
with main_container:
    st.markdown("<h1 style='text-align:center'><i><strong>Juan Martin Vilela </strong></i>&#x1F60E;</h1>",unsafe_allow_html=True)
    st.divider()
    col1, col2 = st.columns([0.65, 0.35],gap='medium')
    with col1:
        st.write(about_short)
    with col2:
        st.image("./perfil.jpg",use_column_width=True)        

ia_container= st.container(border=True)
with ia_container:
    st.subheader("Ask something",anchor=False)
    col3, col4 = st.columns([0.2,0.8], gap='small',vertical_alignment='bottom')
    repuesta=''
    with col3:
        if st.button("Send",use_container_width=True,type='primary',on_click=reset_input('pregunta')):
            texto_introducido=st.session_state['saved-pregunta']
            print("texto introducido",texto_introducido)
            repuesta ="- "+texto_introducido+"\n\n"+ model.generate_content(texto_introducido).text
    with col4:
        st.text_input("pregunta",placeholder="ask the AI ​​something and press send",key="pregunta",label_visibility='hidden')
    st.write(repuesta)
############    
st.divider()
galery_cont=st.container(border=True)
with galery_cont:
    st.write("Galery")
    gal_col1,gal_col2,gal_col3=st.columns(3,gap="medium",vertical_alignment="center")
    with gal_col1:
        st.image("./img/face-detec.jpg")
    with gal_col2:
        st.image("./img/areas.jpg")
        st.image("./img/chica.jfif")
        st.image("./img/background.jpg")
    with gal_col3:
        st.image("./img/foreground.jpg")
        
st.divider()
############
cv2_mediaP=st.container(border=True)
link_format="https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gacbaa02cffc4ec2422dfa2e24412a99e2"
with cv2_mediaP:
    st.write("Handling CV2 & MediaPipe")
    face_detec=st.container(border=True)
    with face_detec:
        st.write("Face Detection ")
        foti=st.file_uploader("Upload a picture")
        cam_col1, cam_col2 = st.columns(2)
        if foti is not None:
            if check_image_buffer(foti.getvalue()) is not -1:
                fotis,foti_detec, resul=face_detect_buffer(foti.getvalue())
                with cam_col1:            
                    st.write("Original")
                    st.image(foti)
                with cam_col2:
                    st.write("Detect")              
                    st.image(foti_detec)
            else:
                st.subheader("Invalid file or format",anchor=None)
                st.page_link(link_format,label="Info")
    segmeter=st.container(border=True)
    with segmeter:
        st.write("Image Segmentation")
        foti_2=st.file_uploader("Upload a picture",key="segmenter")
        if foti_2 is not None:
            if check_image_buffer(foti_2.getvalue()) is not -1:
                seg=imag_segmeter(foti_2.getvalue(),1)
                st.write("Original")
                if foti_2 is not None:
                    st.image(foti_2)
                resul_con=st.container(border=True)
                seg_col1, seg_col2 ,seg_col3 = st.columns(3)        
                with resul_con:
                    with seg_col1:
                        st.write("Foreground")
                        st.image(seg[1])
                    with seg_col2:
                        st.write("Background")
                        st.image(seg[2])
                    with seg_col3:
                        st.write("Areas")
                        st.image(seg[3])
            else:
                st.subheader("Invalid file or format",anchor=None)
                st.page_link(link_format,label="Info")
st.divider()
personal_cont=st.container(border=True)
with personal_cont:
    st.write("Other skills")
    skill=st.container(border=True)
    with skill:
        st.progress(80,"Html, Css an JavaScript skills")
        st.divider()
        st.progress(70,"Python")
        st.progress(30,"Streamlit")
        st.progress(55,"Open CV")
        st.progress(50,"MediaPipe")
        st.progress(50,"Gemini IA")
        st.divider()
        st.progress(60,"Java")
        st.progress(60,"Spring, Spring Boot")
        st.divider()
        st.progress(70,"node - JavaScript")
        st.progress(60,"Express")
        st.progress(60,"Angular 13+")
        st.progress(40,"React")
st.divider()
contact_cont=st.container(border=True)
linkedin_perfil="https://www.linkedin.com/in/juan-martin-vilela-villegas-904bb2238/"
git_page="https://github.com/jmvv"
with contact_cont:
    st.subheader("Contact",anchor=None)
    st.page_link(linkedin_perfil,label="LinkeDin")
    st.page_link(git_page,label="Git")