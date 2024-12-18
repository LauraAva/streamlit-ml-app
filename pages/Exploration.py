import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Data Exploration and Visualizations")

# Retrieve dataset
df = st.session_state.get('data', None)

if df is not None:
    st.write("### Class Distribution")
    target_column = "CO2_class"
    if target_column in df.columns:
        st.bar_chart(df[target_column].value_counts())

    # Plot missing values
    st.write("### Missing Value Statistics")
    st.write(df.isnull().sum())

    # Allow user to select columns to visualize
    st.write("### Column Distribution")
    column_to_plot = st.selectbox("Select a column to plot:", df.columns)
    plt.figure(figsize=(8, 6))
    df[column_to_plot].value_counts().plot(kind="bar")
    plt.title(f"Distribution of {column_to_plot}")
    st.pyplot(plt.gcf())
else:
    st.warning("Please load a dataset first in the 'Dataset Loading' section.")
