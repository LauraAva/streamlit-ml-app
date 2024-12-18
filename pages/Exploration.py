import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Data Exploration and Visualizations")

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
    st.write("### Box Plot")
    if pd.api.types.is_numeric_dtype(df[column_to_plot]):
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=df[column_to_plot], color='lightblue')
        plt.title(f"Box Plot of {column_to_plot}")
        st.pyplot(plt.gcf())

    ### Interactive Dataset Details ###
    st.write("### Dataset Details")
    st.write("Use the filters below to interactively view the dataset.")

    # Column selection
    columns_to_display = st.multiselect("Select columns to display:", df.columns, default=df.columns)

    # Row filter
    rows_to_display = st.slider("Select number of rows to display:", min_value=5, max_value=len(df), value=10)

    # Display filtered dataset
    st.dataframe(df[columns_to_display].head(rows_to_display))

    ### Download Filtered Dataset ###
    st.write("### Download Dataset")
    @st.cache_data
    def convert_df_to_csv(data):
        return data.to_csv(index=False).encode('utf-8')
    
    csv_data = convert_df_to_csv(df[columns_to_display])
    st.download_button(label="Download Dataset", 
                       data=csv_data, 
                       file_name="filtered_data.csv", 
                       mime="text/csv")
else:
    st.warning("Please load a dataset first in the 'Dataset Loading' section.")
