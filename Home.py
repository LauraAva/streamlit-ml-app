import streamlit as st

# Apply custom CSS to clean up the sidebar and remove the extra text
st.markdown("""
    <style>
    /* Remove default sidebar header */
    [data-testid="stSidebar"] h1 {
        display: none;
    }
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
        display: none; /* Hide redundant text above logo */
    }
    /* Sidebar background color and text */
    [data-testid="stSidebar"] {
        background-color: #2b4c7e;
        color: white;
    }
    .st-emotion-cache-16txtl3, .st-emotion-cache-1vd2ayl {
        color: white;
    }
    /* Sidebar navigation buttons */
    .stRadio label {
        color: white;
        font-weight: bold;
        font-size: 16px;
    }
    .stRadio div[role="radiogroup"] > label > span {
        padding-left: 5px;
    }
    /* Sidebar logo styling */
    .stImage img {
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Add the sidebar content
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.title("🔍 CO₂ Emission Analysis")
st.sidebar.markdown("---")  # Separator line for styling

# Sidebar navigation options with icons
pages = [
    "🏠 Home", 
    "📄 Data Loading", 
    "📊 Exploration", 
    "🧪 Modeling", 
    "🔮 Predictions"
]
page_selection = st.sidebar.radio("Go to", pages)

# Navigate between pages
if page_selection == "🏠 Home":
    st.write("## Welcome to the CO₂ Project Home Page!")
elif page_selection == "📄 Data Loading":
    st.experimental_set_query_params(page="Data_Loading")
elif page_selection == "📊 Exploration":
    st.experimental_set_query_params(page="Exploration")
elif page_selection == "🧪 Modeling":
    st.experimental_set_query_params(page="Modeling")
else:
    st.experimental_set_query_params(page="Predictions")

# Main Content for Home Page
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
