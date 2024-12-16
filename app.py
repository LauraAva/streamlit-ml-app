
import streamlit as st
import pandas as pd

# Title of the app
st.title("My First Streamlit App")

# Load and display the dataset
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

# Add a simple chart
st.line_chart(data)

# Add a slider example
slider_value = st.slider("Choose a number", 1, 100, 50)
st.write(f"You selected: {slider_value}")
