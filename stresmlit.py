import streamlit as st
import numpy as np
from PIL import Image
from fastai.vision.all import PILImage, load_learner
from PIL import UnidentifiedImageError
import time

models = {'ResNet34': 'resnet34_model.pkl', 'ResNet50': 'resnet50_model.pkl'}




def image(image_file, logtxtbox):
    try:
        logtxtbox.text_area("Logging: ","Loading the image..", height = 100)
        learn = load_learner(models['ResNet34'])
        time.sleep(10)
        img = PILImage.create(image_file)
        logtxtbox.text_area("Logging: ","PIL Image created ", height = 100)
        time.sleep(10)
        start_time = time.time()
        logtxtbox.text_area("Logging: ","Starting to predict..", height = 100)
        pred, _, probs = learn.predict(img)
        end_time = time.time()
        new_title = f'<p style="font-family:sans-serif; color:Green; font-size: 42px;">Prediction: {pred}; Probability: {max(probs):.04f}; Time taken: {end_time - start_time:.2f} seconds</p>'
        logtxtbox.text_area("Logging :","RESULTS",height = 100)
        st.markdown(new_title, unsafe_allow_html=True)
        
        return True
    except UnidentifiedImageError:
        logtxtbox.text_area('Error: Uploaded file is not a valid image. Please upload a valid image file.',height = 200)
        return False
    except Exception as e:
        logtxtbox.text_area(f'Error: {str(e)}',height = 100)



def main_loop():
    st.title("Tyre Classification App")
    st.subheader("This app allows you to classify the tyre quality")
    st.text("We use OpenCV and Streamlit for this demo")


    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)
    
    st.image([original_image])
    logtxtbox = st.empty()
    logtxt = 'Analysing...'
    logtxtbox.text_area("Logging: ",logtxt, height = 100)
    time.sleep(10)
    image(image_file , logtxtbox)
    st.text("***********COMPLETED***********")

    




if __name__ == '__main__':
    main_loop()