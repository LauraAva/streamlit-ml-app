streamlit run Home.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the app layout and title
st.set_page_config(layout="wide", page_title="CO2 Emission Analysis App")
st.title("CO2 Emission Analysis & Machine Learning Pipeline")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = ["Introduction", "Dataset Loading", "Data Exploration"]
selection = st.sidebar.radio("Go to:", options)

if selection == "Introduction":
    # Landing Page
    st.header("Introduction")
    st.write("""
    Welcome to the CO2 Emission Analysis App! This app allows you to:
    - Explore datasets related to CO2 emissions.
    - Preprocess the data with cleaning, encoding, and scaling steps.
    - Train machine learning models to classify CO2 emission levels.
    - Make predictions on new datasets and download results.

    Navigate through the app using the sidebar.
    """)

elif selection == "Dataset Loading":
    # Dataset Loading Section
    st.header("Dataset Loading")

    # Option to upload a dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV file):", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")
    else:
        # Load a default dataset
        st.info("Using the preloaded dataset.")
        default_url = "https://raw.githubusercontent.com/LauraAva/streamlit-ml-app/refs/heads/main/cl_union_cleaned_BI.csv"
        try:
            df = pd.read_csv(default_url)
            st.success("Default dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error loading the default dataset: {e}")
            st.stop()

    # Display dataset preview
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Display dataset shape
    st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    # Display missing values
    st.write("### Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

elif selection == "Data Exploration":
    # Data Exploration Section
    st.header("Data Exploration")

    # Ensure the dataset is loaded
    if 'df' not in locals():
        st.warning("Please load a dataset first from the 'Dataset Loading' section.")
    else:
        # Summary statistics
        st.write("### Summary Statistics")
        st.write(df.describe())

        # Correlation heatmap
        st.write("### Correlation Heatmap")
        corr = df.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt.gcf())

        # Class distribution for target variable (if present)
        target_column = st.selectbox("Select Target Column for Distribution Analysis:", df.columns)
        if target_column:
            st.write(f"### Class Distribution in {target_column}")
            st.bar_chart(df[target_column].value_counts())
