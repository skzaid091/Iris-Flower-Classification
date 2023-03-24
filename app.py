import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.express as px

data = pd.read_csv('Iris.csv')
data.drop('Id', axis=1, inplace=True)
data.rename(columns={'SepalLengthCm': 'Sepal_Length', 'SepalWidthCm': 'Sepal_Width', 'PetalLengthCm': 'Petal_Length', 'PetalWidthCm': 'Petal_Width'}, inplace=True)

st.set_page_config(layout='wide')
st.sidebar.title('Iris Flower Classification')


def empty_space():
    col4, col5, col6 = st.columns(3)
    with col4:
        st.write('')
        st.write('')
        st.write('')
    with col5:
        pass
    with col6:
        pass


menu = st.sidebar.radio('Select', ('Prediction', 'Visualization'))

if menu == 'Prediction':
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sepal_length = st.text_input('Enter Sepal Length', '0.0', placeholder='Enter numeric value', key='placeholder')
    with col2:
        sepal_width = st.text_input('Enter Sepal Width', '0.0', placeholder='Enter numeric value')
    with col3:
        petal_length = st.text_input('Enter Petal Length', '0.0', placeholder='Enter numeric value')
    with col4:
        petal_width = st.text_input('Enter Petal Width', '0.0', placeholder='Enter numeric value')

    new_data = data.values
    x = new_data[:, 0:4]
    y = new_data[:, 4]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    sepal_length = float(sepal_length)
    sepal_width = float(sepal_width)
    petal_length = float(petal_length)
    petal_width = float(petal_width)

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)

    empty_space()

    btn = st.button(label='Predict')

    empty_space()

    if btn:
        st.subheader('The Predicted Specie is : ' + prediction[0])

if menu == 'Visualization':
    st.title('Sepal Length vs Sepal Width')
    fig1 = px.scatter(data, x='Sepal_Width', y='Sepal_Length', color='Species', symbol='Species')
    st.plotly_chart(fig1, use_container_width=True)

    empty_space()

    st.title('Petal Length vs Petal Width')
    fig2 = px.scatter(data, x='Petal_Width', y='Petal_Length', color='Species', symbol='Species')
    st.plotly_chart(fig2, use_container_width=True)
