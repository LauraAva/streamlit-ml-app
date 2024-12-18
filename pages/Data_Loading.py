import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Loading", page_icon="📄")

st.title("Dataset Loading and Exploration")

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['data'] = df
    st.success("Dataset loaded successfully!")
else:
    # Default Dataset
    st.write("Using preloaded dataset:")
    df = pd.read_csv("cl_union_cleaned_BI.csv")
    st.session_state['data'] = df

# Fix Year column: remove commas and convert to integer
if 'Year' in df.columns:
    df['Year'] = df['Year'].replace({',': ''}, regex=True)  # Remove commas
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')  # Convert to integer
    st.session_state['data'] = df

# Display dataset preview
st.write("### Dataset Preview:")
st.dataframe(df.head())

# Exclude 'Year' column from summary
if 'Year' in df.columns:
    summary_df = df.drop(columns=['Year']).describe()
else:
    summary_df = df.describe()

# Display dataset summary
st.write("### Dataset Summary:")
st.write(summary_df)

# Display missing value statistics
st.write("### Missing Value Statistics:")
st.write(df.isnull().sum())
