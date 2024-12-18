import streamlit as st

# Custom Sidebar Styling
st.markdown("""
    <style>
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #2b4c7e !important;
            color: white !important;
        }

        /* Sidebar title and text */
        [data-testid="stSidebar"] .stTitle,
        [data-testid="stSidebar"] .stText {
            color: #f5f7fa !important;
        }

        /* Remove default Streamlit padding */
        .block-container {
            padding-top: 0;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("logo.png", use_container_width=True)  # Place logo at the top
st.sidebar.title("🔍 CO₂ Emission Analysis")
st.sidebar.markdown("---")  # Separator line

# Navigation Links
pages = ["🏠 Home", "📄 Data Loading", "📊 Exploration", "🧪 Modeling", "🔮 Predictions"]
selected_page = st.sidebar.radio("Go to", pages)

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


