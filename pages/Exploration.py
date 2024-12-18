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

    # Allow user to select columns to visualize
    st.write("### Column Distribution")
    column_to_plot = st.selectbox("Select a column to plot:", df.columns)

    if column_to_plot in df.columns:
        plt.figure(figsize=(8, 6))

        # For numeric columns, use histogram
        if pd.api.types.is_numeric_dtype(df[column_to_plot]):
            plt.hist(df[column_to_plot].dropna(), bins=50, color='skyblue', edgecolor='black')
            plt.title(f"Distribution of {column_to_plot}")
            plt.xlabel(column_to_plot)
            plt.ylabel("Frequency")
        # For categorical columns, use bar plot
        elif df[column_to_plot].nunique() <= 50:  # Avoid plotting high-cardinality columns
            df[column_to_plot].value_counts().plot(kind="bar", color='skyblue')
            plt.title(f"Distribution of {column_to_plot}")
            plt.ylabel("Count")
        else:
            st.warning(f"'{column_to_plot}' has too many unique values to visualize as a barplot.")

        st.pyplot(plt.gcf())
else:
    st.warning("Please load a dataset first in the 'Dataset Loading' section.")
