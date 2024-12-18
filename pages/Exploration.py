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
            # Numeric columns: Use Histogram
            bins = st.slider("Select number of bins", min_value=5, max_value=100, value=30)
            plt.hist(df[column_to_plot].dropna(), bins=bins, color='skyblue', edgecolor='black')
            plt.title(f"Distribution of {column_to_plot}")
            plt.xlabel(column_to_plot)
            plt.ylabel("Frequency")
            st.pyplot(plt.gcf())
        elif df[column_to_plot].nunique() <= 10:
            # Categorical Columns: Pie Chart
            pie_data = df[column_to_plot].value_counts()
            fig, ax = plt.subplots()
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, 
                   colors=sns.color_palette('pastel'), textprops={'fontsize': 8})
            plt.title(f"Proportion of {column_to_plot}")
            st.pyplot(fig)
        elif df[column_to_plot].nunique() <= 50:
            # Categorical columns: Bar Plot
            df[column_to_plot].value_counts().plot(kind="bar", color='skyblue')
            plt.title(f"Distribution of {column_to_plot}")
            plt.ylabel("Count")
            st.pyplot(plt.gcf())
        else:
            st.warning(f"'{column_to_plot}' has too many unique values for a barplot.")

    ### Box Plot ###
    st.write("### Box Plot")
    if pd.api.types.is_numeric_dtype(df[column_to_plot]):
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=df[column_to_plot], color='lightblue')
        plt.title(f"Box Plot of {column_to_plot}")
        st.pyplot(plt.gcf())

    ### Interactive Dataset Filtering ###
    st.write("### Interactive Dataset Filtering")
    filter_col = st.selectbox("Select column to filter:", df.columns)
    if pd.api.types.is_numeric_dtype(df[filter_col]):
        min_val, max_val = st.slider(f"Filter {filter_col} range", 
                                     float(df[filter_col].min()), float(df[filter_col].max()), 
                                     (float(df[filter_col].min()), float(df[filter_col].max())))
        filtered_df = df[(df[filter_col] >= min_val) & (df[filter_col] <= max_val)]
    else:
        unique_vals = df[filter_col].unique()
        selected_vals = st.multiselect(f"Select values for {filter_col}", unique_vals, default=unique_vals[:5])
        filtered_df = df[df[filter_col].isin(selected_vals)]
    st.write("### Filtered Dataset")
    st.dataframe(filtered_df.head(10))

    ### Correlation Heatmap ###
    st.write("### Correlation Heatmap")
    numeric_cols = df.select_dtypes(include='number')
    if not numeric_cols.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for correlation heatmap.")

    ### Download Cleaned/Filtered Data ###
    st.write("### Download Filtered Dataset")
    @st.cache_data
    def convert_df_to_csv(data):
        return data.to_csv(index=False).encode('utf-8')
    
    csv_data = convert_df_to_csv(filtered_df)
    st.download_button(label="Download Filtered Data", 
                       data=csv_data, 
                       file_name="filtered_data.csv", 
                       mime="text/csv")
else:
    st.warning("Please load a dataset first in the 'Dataset Loading' section.")
