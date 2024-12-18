# Custom Sidebar for Navigation
st.sidebar.image("logo.png", use_container_width=True)  # Updated to use_container_width
st.sidebar.title("🔍 CO₂ Emission Analysis")
st.sidebar.markdown("---")  # Separator line

# Sidebar navigation options
st.sidebar.write("### Navigation")
pages = ["🏠 Home", "📄 Data Loading", "📊 Exploration", "🧪 Modeling", "🔮 Predictions"]
page_selection = st.sidebar.radio("Go to", pages)

# Navigate between pages using query parameters
if page_selection == "🏠 Home":
    st.query_params.update({"page": "Home"})  # Updated to st.query_params
elif page_selection == "📄 Data Loading":
    st.query_params.update({"page": "Data_Loading"})
elif page_selection == "📊 Exploration":
    st.query_params.update({"page": "Exploration"})
elif page_selection == "🧪 Modeling":
    st.query_params.update({"page": "Modeling"})
else:
    st.query_params.update({"page": "Predictions"})

# Main Content
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
