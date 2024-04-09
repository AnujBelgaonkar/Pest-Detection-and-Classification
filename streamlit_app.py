import os
import pandas as pd
import together
import streamlit as st
import numpy as np
from PIL import Image
from resources import get_model
import streamlit_scrollable_textbox as stx
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
load_dotenv(find_dotenv())

os.environ["TOGETHER_API_TOKEN"] =os.getenv("TOGETHER_API_TOKEN") 

path = r"model.weights.h5"

def prompt(text):
    output = together.Complete.create(
        prompt=text,
        model="OPENCHAT/OPENCHAT-3.5-1210",
        max_tokens=500,
        temperature=0.5,
    )
    response = output["output"]["choices"][0]
    return response

def preprocess(image) -> np:
    image = image.resize((224, 224))
    img_array = np.asarray(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array  

st.set_page_config(layout='wide')

def main():

    model = get_model()
    model.load_weights(path)
    img_array = None
    names = {0: 'Ants', 1: 'Bees', 2: 'Bettle', 3: 'Cattterpillar', 4: 'Earthworms',
             5: 'Earwig', 6: 'Grasshopper', 7: 'Moth', 8: 'Slug', 9: 'Snail', 10: 'Wasp', 11: 'Weevil'}

    header = st.container()
    image_column, predict, prediction_column = st.columns((2, 1, 2), gap='large')

    with header:
        st.write(
            """
            <div style="display: flex; justify-content: center;">
                <h1>Pest Detection Web App</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with image_column:
        upload = st.file_uploader("Upload Image", type=['png', 'jpg'])
        if upload is not None:
            image = Image.open(upload)
            st.image(image)
            img_array = preprocess(image)

    with predict:
        pressed = st.button("Predict", key='predict' "primary", use_container_width=True)
            
    with prediction_column:
        st.header("Prediction")
        if pressed:
            if img_array is not None:
                x = model.predict(img_array)
                result = np.argmax(x)
                pest = names.get(result)
                st.text(f"The predicted pest is {pest}")
                text = f"What are the best agricultural practices to deal with {pest}. What practicies should a farmer use"
                query = prompt(text)
                stx.scrollableTextbox(text=query["text"], height=400, border=True)

    with st.sidebar:
        records = st.container()

if __name__ == "__main__": 
    main()
