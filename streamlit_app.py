import streamlit as st
import numpy as np
from PIL import Image
#import tensorflow as tf
#from tensorflow import keras
#from keras.models import Model
from resources import get_model,load_llm
from langchain.llms import Ollama
import streamlit_scrollable_textbox as stx
import os
import replicate
from dotenv import load_dotenv,find_dotenv


load_dotenv(find_dotenv())



os.environ["REPLICATE_API_TOKEN"] =os.getenv("REPLICATE_API_TOKEN") 
path = r"model.weights.h5"

def prompt(text,pre_prompt,api : bool):
    if api == True:
        output = replicate.run(
            "meta/llama-2-7b",
            
            input = {
                "prompt": text,
                "system_prompt" : "You are an assistant not an enviromentalist answer as asked.",
                "max_new_tokens" : 500
            })
        return ''.join(output)
    else:
        local_llm = load_llm()
        return local_llm(text)
    

def preprocess(image) -> np:
    image = image.resize((224,224))
    img_array = np.asarray(image)
    img_array = img_array/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array    

model = get_model()
model.load_weights(path)
img_array = None
names = {0 : 'Ants', 1 : 'Bees', 2:'Bettle',3:'Cattterpillar',4:'Earthworms',
         5:'Earwig',6:'Grasshopper', 7:'Moth',8:'Slug',9:'Snail',10:'Wasp',11:'Weevil'}


header = st.container()
image_column,prediction_column = st.columns(2,gap = 'large')

with header:
    st.title("  Pest Detection Web App  ")

with st.sidebar:
    records = st.container()
    


with image_column:
    upload = st.file_uploader("Upload Image", type = ['png','jpg'])
    if upload is not None:
        image = Image.open(upload)
        st.image(image)
        img_array = preprocess(image)
        
with prediction_column:
    st.header("Prediction")
    if img_array is not None:
        x = model.predict(img_array)
        result = np.argmax(x)
        pest = names.get(result)
        st.text(f"The predicted pest is {pest}")
        pre_prompt = "You are a helpful assistant. Give precise answers."
        text = f"What are the best agricultural practices to deal with {pest}. What practicies should a farmer +"
        query = prompt(text,pre_prompt,False)
        #answer = response(query)
        stx.scrollableTextbox(text = query, height = 400, border = True)


