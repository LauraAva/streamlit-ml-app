import streamlit as st
# Global CSS Styling
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background-color: #f5f7fa;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #2b4c7e;
        color: white;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #f5f7fa;
    }
    /* Title styling */
    .stTitle {
        color: #2b4c7e;
        font-weight: bold;
        text-align: center;
    }
    /* Buttons */
    .stButton>button {
        background-color: #2b4c7e;
        color: white;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Home", page_icon="🏠")
# Custom Sidebar for Navigation
st.sidebar.image("your_logo.png", use_column_width=True)  # Add your project logo
st.sidebar.title("🔍 CO₂ Emission Analysis")
st.sidebar.markdown("---")  # Separator line

# Sidebar navigation options
st.sidebar.write("### Navigation")
pages = ["🏠 Home", "📄 Data Loading", "📊 Exploration", "🧪 Modeling", "🔮 Predictions"]
page_selection = st.sidebar.radio("Go to", pages)

# Navigate between pages using query parameters
if page_selection == "🏠 Home":
    st.experimental_set_query_params(page="Home")
elif page_selection == "📄 Data Loading":
    st.experimental_set_query_params(page="Data_Loading")
elif page_selection == "📊 Exploration":
    st.experimental_set_query_params(page="Exploration")
elif page_selection == "🧪 Modeling":
    st.experimental_set_query_params(page="Modeling")
else:
    st.experimental_set_query_params(page="Predictions")

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
