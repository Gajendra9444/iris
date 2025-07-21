import streamlit as st
import pandas as pd
import numpy as np
from prediction import predictor


st.title('Classifying Tris Flower')
st.markdown('Toy to play to calssify iris flowers into\
            (setosa, versicolor, virginica) based on their sepal/petal\
                and leangth/weidth. ')

st.header("Plant Features")
col1, col2 = st.columns(2)


with col1:
    st.text("Sepal characterstics")
    sepal_l = st.slider('Sepal length (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 0.1, 2.5, 0.5)


with col2:
    st.text("Petal characterstics")
    petal_l = st.slider('Petal length (cm)', 1.0, 8.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

st.text('')
if st.button("Predict type of Iris"):
    result = predictor(np.array([[sepal_l, sepal_w,petal_l,petal_w]]))
    predicted_class = result[0]
    st.success(f"Predicted: {predicted_class.title()}")


    image_path = f"image/{predicted_class.lower()}.jpg"
    st.image(image_path, caption=predicted_class.title(), use_container_width=True)


st.text('')
st.text('')
st.markdown(
    '`created by` Gajendra')    