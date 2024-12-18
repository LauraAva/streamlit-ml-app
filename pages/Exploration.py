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

import seaborn as sns

# Add a heatmap for numeric correlations
st.write("### Correlation Heatmap")
numeric_cols = df.select_dtypes(include='number')
if not numeric_cols.empty:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
else:
    st.warning("No numeric columns available for correlation heatmap.")

# Add bin control for numeric distributions
if pd.api.types.is_numeric_dtype(df[column_to_plot]):
    st.write(f"### Distribution of {column_to_plot}")
    bins = st.slider("Select number of bins", min_value=5, max_value=100, value=30)
    plt.figure(figsize=(8, 6))
    plt.hist(df[column_to_plot].dropna(), bins=bins, color='lightblue', edgecolor='black')
    plt.title(f"Distribution of {column_to_plot} with {bins} bins")
    plt.xlabel(column_to_plot)
    plt.ylabel("Frequency")
    st.pyplot(plt.gcf())

st.write("### Filter Dataset")

# Example: Filter numeric column
if column_to_plot in numeric_cols.columns:
    min_val, max_val = st.slider(f"Filter {column_to_plot}", 
                                 float(df[column_to_plot].min()), 
                                 float(df[column_to_plot].max()), 
                                 (float(df[column_to_plot].min()), float(df[column_to_plot].max())))
    filtered_df = df[(df[column_to_plot] >= min_val) & (df[column_to_plot] <= max_val)]
    st.write(filtered_df.head(10))
else:
    # Filter categorical columns
    unique_vals = df[column_to_plot].unique()
    selected_vals = st.multiselect(f"Filter {column_to_plot}", unique_vals, default=unique_vals[:5])
    filtered_df = df[df[column_to_plot].isin(selected_vals)]
    st.write(filtered_df.head(10))

st.write("### Box Plot")
if pd.api.types.is_numeric_dtype(df[column_to_plot]):
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[column_to_plot], color='lightblue')
    plt.title(f"Box Plot of {column_to_plot}")
    st.pyplot(plt.gcf())

if pd.api.types.is_numeric_dtype(df[column_to_plot]):
    st.write(f"### Summary Statistics for {column_to_plot}")
    st.write(df[column_to_plot].describe())

st.write("### Pie Chart")
if column_to_plot in df.columns and df[column_to_plot].nunique() <= 10:
    pie_data = df[column_to_plot].value_counts()
    fig, ax = plt.subplots()
    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    plt.title(f"Proportion of {column_to_plot}")
    st.pyplot(fig)
st.write("### Dataset Details")
show_data = st.checkbox("Show full dataset", value=False)
if show_data:
    st.dataframe(df)
else:
    st.write("Showing first 10 rows:")
    st.dataframe(df.head(10))
