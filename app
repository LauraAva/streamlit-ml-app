1. Directory Structure
/app/
Home.py: Introductory page.
Data_Loading.py: Handles dataset uploading and display.
Exploration.py: Dataset analysis and exploration.
Modeling.py: Model training and evaluation.
Predictions.py: Handles prediction using new CSV uploads.

import streamlit as st

st.title("CO2 Emission Analysis & Machine Learning Pipeline")
st.write("""
Welcome to the CO2 Emission Analysis app. Navigate using the sidebar to:
1. **Load Dataset**
2. **Explore the Data**
3. **Train Models**
4. **Make Predictions**
""")
import streamlit as st
import pandas as pd

st.title("Load Dataset")

st.write("""
Upload a dataset to begin your analysis. You can use the default dataset if none is uploaded.
""")

# Default Dataset
url = "https://raw.githubusercontent.com/LauraAva/streamlit-ml-app/refs/heads/main/cl_union_cleaned_BI.csv"
df = None

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

try:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")
    else:
        df = pd.read_csv(url)
        st.info("Using default dataset.")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Error loading dataset: {e}")
import streamlit as st

st.title("Data Exploration")

st.write("Explore your dataset here.")
# Placeholder for DataFrame display or visualizations
import streamlit as st

st.title("Predictions")

st.write("Upload a CSV file to make predictions based on your trained model.")
# Placeholder for prediction logic
