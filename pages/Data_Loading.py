import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Loading", page_icon="📄")

st.title("Dataset Loading and Exploration")

# Ensure session state is initialized
if 'data' not in st.session_state:
    st.session_state['data'] = None  # Initialize session state data

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['data'] = df  # Save dataset to session state
    st.success("Dataset loaded successfully!")
else:
    # Default Dataset
    st.write("Using preloaded dataset:")
    df = pd.read_csv("cl_union_cleaned_BI_combined_file.csv")
    st.session_state['data'] = df  # Save default dataset to session state

# Display Dataset Preview
if st.session_state['data'] is not None:
    st.write("### Dataset Preview:")
    st.dataframe(st.session_state['data'].head())
    st.write("### Dataset Information:")
    st.write('''
    ## CL Dataset
    The French government website or ADEME provides annual data from the Union Technique de l'Automobile du Motorcycle et du Cycle (UTAC), the organization responsible for vehicle approval before they are released for sale, in collaboration with the Ministry of Sustainable Development since 2001 (Europa.eu, 2023). The original data provided by UTAC for each vehicle includes:
    - Fuel consumption
    - Carbon dioxide (CO2) emissions
    - Emissions of air pollutants (regulated under the Euro standards)
    - All technical specifications of the vehicles (such as ranges, brands, models, CNIT number, energy type, etc.)
    - The values for the bonus-malus system and the Energy Class - CO2 label (which vary according to regulations derived from the Finance Law and its decrees)

    The dataset has gathered information from 2001 until 2015 in csv files. Due to the vast amount of data and the limited time frame datasets from 2012 to 2014 have been used to conduct the project. The 2012 file has a memory of 464 KB, the 2013 file has a memory of 479 KB and the size of the 2014 file is 578 KB. The datasets are currently named CL_2012, CL_2013 and CL_2014. The datasets include the manufacturer and specific model of each vehicle. This is crucial for identifying which vehicles are associated with higher emissions. Furthermore, information about the fuel type (e.g., petrol, diesel, hybrid, electric) is included as well as engine size, power output (Kw), curb weight and other technical characteristics.

    The datasets contain values for CO2 emissions, measured in grams per kilometre (g/km). This is the primary metric for evaluating the environmental impact of the vehicles. Further, other pollutants such as NOx (Nitrogen Oxides), which are harmful gases contributing to air pollution and respiratory problems, HC (Hydrocarbons), organic compounds that contribute to smog and ground-level ozone, and particles that can penetrate deep into the lungs and cause health issues have been provided. Lastly, fuel consumption under different driving conditions, such as urban, extra-urban and mixed, is another valuable variable in the dataset and measured in litres per 100 kilometres (l/100km). 

''')
    
# Fix Year column: remove commas and convert to integer
if 'Year' in df.columns:
    df['Year'] = df['Year'].replace({',': ''}, regex=True)  # Remove commas
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')  # Convert to integer
    st.session_state['data'] = df

# Display Dataset Summary
    # Exclude 'Year' column from summary
    if 'Year' in st.session_state['data'].columns:
        summary_df = st.session_state['data'].drop(columns=['Year']).describe()
    else:
        summary_df = st.session_state['data'].describe()

    st.write("### Dataset Summary:")
    st.write(summary_df)

    # Display missing value statistics
    st.write("### Missing Value Statistics:")
    missing_values = st.session_state['data'].isnull().sum().reset_index()
    missing_values.columns = ['Column', 'Missing Values']
    st.dataframe(missing_values)

