import os
import pandas as pd
import together
import streamlit as st
import numpy as np
from PIL import Image
from resources import get_model
import streamlit_scrollable_textbox as stx
from dotenv import load_dotenv, find_dotenv
from langchain_together import Together

load_dotenv(find_dotenv())

#os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]
os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")

def init_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = []
    if 'counter' not in st.session_state:
        st.session_state.counter = 1

path = r"model.weights.h5"

def add_data(data):
    st.session_state.data.append((st.session_state.counter, data))
    st.session_state.counter += 1

@st.cache_resource
def get_llm():
    llm = Together(
        model="META-LLAMA/LLAMA-2-7B-CHAT-HF",
        max_tokens=700,
        temperature=0.6,
        top_p= 0.15
    )
    return llm

def prompt(text):
    llm = get_llm()
    output = llm.invoke(text)
    return output

def preprocess(image) -> np:
    image = image.resize((224, 224))
    img_array = np.asarray(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array  

st.set_page_config(layout='wide')

def main():
    init_session_state()
    model = get_model()
    model.load_weights(path)
    img_array = None
    names = {0: 'Ants', 1: 'Bees', 2: 'Bettle', 3: 'Cattterpillar', 4: 'Earthworms',
             5: 'Earwig', 6: 'Grasshopper', 7: 'Moth', 8: 'Slug', 9: 'Snail', 10: 'Wasp', 11: 'Weevil'}

    header = st.container()
    image_column, prediction_column = st.columns(2, gap='large')
    predict = st.container()

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
        st.write("<br>", unsafe_allow_html=True)  # Add some space above the button
        button_container = st.container()
        with button_container:
            st.markdown("<br>", unsafe_allow_html=True)  # Add some space above the button
            pressed = st.button("Predict", key='predict', use_container_width=False)

            
    with prediction_column:
        st.header("Prediction")
        if pressed:
            if img_array is not None:
                x = model.predict(img_array)
                result = np.argmax(x)
                pest = names.get(result)
                add_data(pest)
                st.text(f"The predicted pest is {pest}")
                text = f"What are the best agricultural practices to deal with {pest}. What practicies should a farmer use"
                query = prompt(text)
               # st.write(query)
                stx.scrollableTextbox(text=query, height=400, border=True)

    st.sidebar.title("Records")
    for idx, data in st.session_state.data:
        st.sidebar.markdown(f"<h3>{idx}: {data}</h3>", unsafe_allow_html=True)
            

if __name__ == "__main__": 
    main()
