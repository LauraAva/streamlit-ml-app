import streamlit as st

# Global CSS for styling
st.markdown("""
    <style>
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #2b4c7e !important; /* Blue background */
        color: white !important; /* White text */
    }
    [data-testid="stSidebar"] .css-1d391kg, 
    [data-testid="stSidebar"] h1, h2, h3, {
        color: white !important; /* Make sidebar text white */
    }
    /* Logo and spacing */
    .sidebar-logo {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    /* Navigation Radio Button Styling */
    .stRadio > div {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    "<div class='sidebar-logo'><img src='https://path_to_your_logo/logo.png' width='100%'></div>", 
    unsafe_allow_html=True
)

# Sidebar Navigation
st.sidebar.image("logo.png", use_container_width=True)  # Logo at the top
st.sidebar.title("🔍 CO₂ Emission Analysis")
st.sidebar.markdown("---")  # Separator line

# Sidebar Navigation Links
pages = {
    "🏠 Home": "Home",
    "📄 Data Loading": "Data_Loading",
    "📊 Exploration": "Exploration",
    "🧪 Modeling": "Modeling",
    "🔮 Predictions": "Predictions"
}

page_selection = st.sidebar.radio("Go to", list(pages.keys()))
# Redirect to selected page using query params
st.query_params.update({"page": pages[page_selection]})

# Conditional navigation
if page_selection == "🏠 Home":
    st.query_params.update({"page": "Home"})
elif page_selection == "📄 Data Loading":
    st.query_params.update({"page": "Data_Loading"})
elif page_selection == "📊 Exploration":
    st.query_params.update({"page": "Exploration"})
elif page_selection == "🧪 Modeling":
    st.query_params.update({"page": "Modeling"})
else 
    st.query_params.update({"page": "Predictions"})


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
