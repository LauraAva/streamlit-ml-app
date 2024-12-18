import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Loading", page_icon="📄")

st.title("Dataset Loading and Exploration")

# Ensure session state is initialized
if 'data' not in st.session_state:
    st.session_state['data'] = None  # Initialize session state data

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['data'] = df  # Save dataset to session state
    st.success("Dataset loaded successfully!")
else:
    # Default Dataset
    st.write("Using preloaded dataset:")
    df = pd.read_csv("cl_union_cleaned_BI_combined_file.csv")
    st.session_state['data'] = df  # Save default dataset to session state

# Fix Year column: remove commas and convert to integer
if 'Year' in df.columns:
    df['Year'] = df['Year'].replace({',': ''}, regex=True)  # Remove commas
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')  # Convert to integer
    st.session_state['data'] = df

# Display Dataset Summary
if st.session_state['data'] is not None:
    st.write("### Dataset Preview:")
    st.dataframe(st.session_state['data'].head())

    # Exclude 'Year' column from summary
    if 'Year' in st.session_state['data'].columns:
        summary_df = st.session_state['data'].drop(columns=['Year']).describe()
    else:
        summary_df = st.session_state['data'].describe()

    st.write("### Dataset Summary:")
    st.write(summary_df)

    # Display missing value statistics
    st.write("### Missing Value Statistics:")
    st.write(st.session_state['data'].isnull().sum())
