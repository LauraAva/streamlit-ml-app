import streamlit as st

# Custom Styling to Fix Header Cut-Off and Sidebar
st.markdown("""
    <style>
        /* Sidebar Background */
        [data-testid="stSidebar"] {
            background-color: #2b4c7e !important;
        }

        /* Sidebar Text Styling */
        [data-testid="stSidebar"] * {
            color: #f5f7fa !important;
        }

        /* Sidebar Title Styling */
        [data-testid="stSidebar"] h1 {
            color: #ffffff !important;
            font-size: 20px !important;
            font-weight: bold !important;
            margin-top: 20px !important;
        }

        /* Main Title and Content Spacing */
        .block-container {
            padding-top: 2rem !important;
        }
        
        /* Adjust Header Position */
        h1, h2, h3 {
            margin-top: 0 !important;
            padding-top: 0.5rem !important;
        }

        /* Optional: Logo Alignment */
        [data-testid="stSidebar"] img {
            display: block;
            margin: 0 auto 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Logo
st.sidebar.image("logo.png", use_container_width=True)

# Title for the Sidebar
st.sidebar.title("🔍 CO₂ Emission Analysis")
st.sidebar.markdown("---")  # Separator line

# Header Content
st.title("CO₂ Emission Analysis & Prediction Pipeline")
st.write("""
## Welcome to the App 🚗  
This app allows you to:
1. **Explore your dataset** 📊  
2. **Train machine learning models** 🤖  
3. **Make predictions on new data** 🗂  
4. **Download results** 📥  

## How to Use:
- Use the sidebar to navigate through different sections.
- Follow the workflow for seamless analysis.
- Upload your dataset or use the provided one.

## Dataset:
- Preloaded Dataset: `cl_union_cleaned_BI_combined_file.csv`
- Target Variable: CO₂ Emission Classes (A to G)

Navigate through the sections using the sidebar! 👈  
""")


