import os
import together
import streamlit as st
import numpy as np
import tensorflow
from PIL import Image
import streamlit_scrollable_textbox as stx
from dotenv import load_dotenv, find_dotenv
from langchain_together import Together
from deep_translator import GoogleTranslator
load_dotenv(find_dotenv())

st.set_page_config(layout='wide')
def init_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = []
    if 'counter' not in st.session_state:
        st.session_state.counter = 1

os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]
#os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")

def add_data(data):
    st.session_state.data.append((st.session_state.counter, data))
    st.session_state.counter += 1

@st.cache_resource
def get_llm():
    llm = Together(
        model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        max_tokens=1000,
        temperature=0.7,
        top_p= 0.7,
        top_k=50,
        repetition_penalty=1
    )
    return llm

def get_response(text):
    llm = get_llm()
    response = llm.invoke(text)
    return response

@st.cache_data
def translate_to(word,end):
    translator = GoogleTranslator(source='auto', target=end)
    return translator.translate(word)

def preprocess(image) -> np:
    image = image.resize((224, 224))
    img_array = np.asarray(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array  


def predict_insect(model,image):
    x = model.predict(image)
    result = np.argmax(x)
    confidence_score = x[0][result]
    return result,confidence_score

@st.cache_resource
def load_model():
    return tensorflow.keras.models.load_model('custom_model26.keras')


def main():
    init_session_state()
    model = load_model()
    img_array = None
    names = {0: 'Ants', 1: 'Bees', 2: 'Bettle', 3: 'Cattterpillar', 4: 'Earthworms',
             5: 'Earwig', 6: 'Grasshopper', 7: 'Moth', 8: 'Slug', 9: 'Snail', 10: 'Wasp', 11: 'Weevil'
             }

    header = st.container()
    image_column, empty ,prediction_column = st.columns((2,1,2), gap='large')
    predict = st.container()
    button = st.container()
    with header:
        st.write(
    """
    <div style="display: flex; justify-content: center;">
        <h1>🐌 Pest Detection Web App 🌱</h1>
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
            with button:
                 pressed = st.button("Predict", key='predict', use_container_width=False)
            language = st.radio(label="Language", options=["English","Hindi"], index=0,horizontal=True)
    with empty:
        ##fill space
        st.write("     ")
            
    with prediction_column:
        st.header("Prediction")
        if pressed:
            if img_array is not None:
                result,confidence_score = predict_insect(model,img_array)
                if confidence_score>0.5:
                    confidence_score = round(confidence_score,2)
                    pest = names.get(result)
                    add_data(pest)
                    st.text(f"The predicted pest is {pest} with a confidence of {confidence_score}")
                    text = [f"What are the best agricultural practices to deal with {pest}. What practicies should a farmer use?",
                            " से निपटने के लिए सर्वोत्तम कृषि पद्धतियाँ क्या हैं? एक किसान को कौन सी पद्धतियों का उपयोग करना चाहिए?",
                            " चा सामना करण्यासाठी सर्वोत्तम कृषी पद्धती कोणत्या आहेत. शेतकऱ्याने कोणत्या पद्धती वापरल्या पाहिजेत?"]
                    if language == "English":
                        st.write(text[0])
                        stx.scrollableTextbox(get_response(text[0]),height= 300)
                    elif language == "Hindi":
                        kida = translate_to(pest,'hi')
                        st.write(kida + text[1])
                        stx.scrollableTextbox(get_response(kida + text[1]),height= 300)
                else:
                    st.write("The model is not confident enough to answer.")

    st.sidebar.title("Records")
    for idx, data in st.session_state.data:
        st.sidebar.markdown(f"<h3>{idx}: {data}</h3>", unsafe_allow_html=True)
            

if __name__ == "__main__": 
    main()
