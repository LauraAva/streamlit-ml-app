import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏠")

st.title("CO2 Emission Analysis & Prediction Pipeline")
st.write("""
## Project Overview
Welcome to the CO2 Emission Analysis and Prediction App! 🚗  
This app enables you to:
1. **Explore and clean datasets** 📊  
2. **Build machine learning models** 🤖  
3. **Make predictions on new data** 🗂  
4. **Download processed data and results** 📥  

## How to Use:
- Use the sidebar to navigate through different sections.
- Follow the workflow for seamless analysis.
- Upload your dataset or use the provided one.

## Dataset:
- Preloaded Dataset: `cl_union_cleaned_BI_combined_file.csv`
- Target Variable: CO2 Emission Classes (A to G)

Navigate through the sections using the sidebar! 👈  
""")
