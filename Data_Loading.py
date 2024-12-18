import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Loading", page_icon="📄")

st.title("Load Dataset")

st.write("Upload a dataset to begin your analysis. If none is uploaded, the default dataset will be used.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
else:
    url = "https://raw.githubusercontent.com/LauraAva/streamlit-ml-app/refs/heads/main/cl_union_cleaned_BI.csv"
    df = pd.read_csv(url)
    st.info("Using the default dataset.")

st.write("Preview of the Dataset:")
st.dataframe(df.head())
