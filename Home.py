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
        padding-top: 20px;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        display: none;  /* Hide default headers */
    }
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
        display: none;  /* Hide the redundant markdown container */
    }
    /* Sidebar text and links */
    .stRadio label {
        color: white;
        font-size: 16px;
        font-weight: bold;
    }

    /* Sidebar logo */
    .stImage img {
        margin-top: -20px;
        margin-bottom: 20px;
        border-radius: 10px;
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

# Sidebar Content with Icons and Navigation
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.title("🔍 CO₂ Emission Analysis")
st.sidebar.markdown("---")  # Separator line

# Sidebar navigation options
st.sidebar.write("### Navigation")
pages = [
    "🏠 Home", 
    "📄 Data Loading", 
    "📊 Exploration", 
    "🧪 Modeling", 
    "🔮 Predictions"
]
page_selection = st.sidebar.radio("Go to", pages)

# Page Navigation
if page_selection == "🏠 Home":
    st.query_params["page"] = "Home"
elif page_selection == "📄 Data Loading":
    st.query_params["page"] = "Data_Loading"
elif page_selection == "📊 Exploration":
    st.query_params["page"] = "Exploration"
elif page_selection == "🧪 Modeling":
    st.query_params["page"] = "Modeling"
else:
    st.query_params["page"] = "Predictions"

# Main Content
st.title("CO₂ Emission Analysis & Prediction Pipeline")
st.write("""
## Project Overview
Welcome to the CO₂ Emission Analysis and Prediction App! 🚗  
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
- Target Variable: CO₂ Emission Classes (A to G)

Navigate through the sections using the sidebar! 👈  
""")
