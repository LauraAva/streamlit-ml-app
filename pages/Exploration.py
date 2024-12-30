import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import setup_sidebar

# Set up the sidebar
setup_sidebar()

st.title("📊 Data Exploration & Vizualisations")
st.write("Upload your dataset here or use the preloaded dataset.")

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['data'] = df
    st.success("Dataset loaded successfully!")
else:
    st.warning("Please upload a dataset.")


# Retrieve dataset
df = st.session_state.get('data', None)

if df is not None:
    ### Class Distribution ###
    st.write("### Class Distribution")
    target_column = "CO2_class"
    if target_column in df.columns:
        st.bar_chart(df[target_column].value_counts())

    ### Column Visualization ###
    st.write("### Column Distribution")
    column_to_plot = st.selectbox("Select a column to plot:", df.columns)

    if column_to_plot in df.columns:
        plt.figure(figsize=(8, 6))
        if pd.api.types.is_numeric_dtype(df[column_to_plot]):
            # Numeric columns: Use Histogram with bin slider
            bins = st.slider("Select number of bins", min_value=5, max_value=100, value=30, key="bin_slider")
            plt.hist(df[column_to_plot].dropna(), bins=bins, color='skyblue', edgecolor='black')
            plt.title(f"Distribution of {column_to_plot}")
            plt.xlabel(column_to_plot)
            plt.ylabel("Frequency")
            st.pyplot(plt.gcf())
        elif df[column_to_plot].nunique() <= 50:
            # Categorical columns: Bar Plot
            df[column_to_plot].value_counts().plot(kind="bar", color='skyblue')
            plt.title(f"Distribution of {column_to_plot}")
            plt.ylabel("Count")
            st.pyplot(plt.gcf())
        else:
            st.warning(f"'{column_to_plot}' has too many unique values for a barplot.")

    ### Correlation Heatmap ###
    st.write("### Correlation Heatmap")
    numeric_cols = df.select_dtypes(include='number').columns
    selected_cols = st.multiselect("Select columns for correlation heatmap:", numeric_cols, default=list(numeric_cols))
    if selected_cols:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[selected_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please select at least one column to display the correlation heatmap.")

    ### Box Plot ###
   ### Box Plot ###
st.write("### Box Plot")
if 'data' in st.session_state and st.session_state['data'] is not None:
    df = st.session_state['data']
    numeric_columns = df.select_dtypes(include='number').columns

    if len(numeric_columns) > 0:
        column_to_plot = st.selectbox("Select a numeric column for the box plot:", numeric_columns)

        if pd.api.types.is_numeric_dtype(df[column_to_plot]):
            plt.figure(figsize=(8, 6))
            sns.boxplot(y=df[column_to_plot], palette="Blues")
            plt.title(f"Box Plot of {column_to_plot}")
            plt.tight_layout()
            st.pyplot(plt.gcf())
    else:
        st.warning("No numeric columns available for box plot.")
else:
    st.warning("Please upload a dataset to display the box plot.")


    ### Interactive Dataset Details ###
    st.write("### Dataset Details")
    st.write("Use the filters below to interactively view the dataset.")

    # Column selection
    columns_to_display = st.multiselect("Select columns to display:", df.columns, default=df.columns)

    # Row filter
    rows_to_display = st.slider("Select number of rows to display:", min_value=5, max_value=len(df), value=10)

    # Sorting options
    sort_column = st.selectbox("Select a column to sort by:", columns_to_display)
    sort_order = st.radio("Select sorting order:", ["Ascending", "Descending"], index=0)

    # Apply sorting
    if sort_column:
        sorted_df = df.sort_values(by=sort_column, ascending=(sort_order == "Ascending"))
    else:
        sorted_df = df

    # Display filtered and sorted dataset
    st.dataframe(sorted_df[columns_to_display].head(rows_to_display))

# Check if dataset is loaded
if 'data' in st.session_state and st.session_state['data'] is not None:
    df = st.session_state['data']
    columns_to_display = st.multiselect("Select columns to display:", df.columns, default=df.columns)

    ### Download Filtered Dataset ###
    st.write("### Download Dataset")
    @st.cache_data
    def convert_df_to_csv(data):
        return data.to_csv(index=False).encode('utf-8')
    
    csv_data = convert_df_to_csv(df[columns_to_display])
    st.download_button(
        label="Download Dataset", 
        data=csv_data, 
        file_name="filtered_data.csv", 
        mime="text/csv"
    )
else:
    st.warning("Please load a dataset first in the 'Dataset Loading' section.")
