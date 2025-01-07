import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Title
st.title("CO2 Emission Time Series Analysis")

# Upload Dataset
st.header("Upload CO2 Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    st.subheader("Preprocessing the Dataset")
    df['year'] = pd.to_datetime(df['year'], errors='coerce')
    df = df.dropna(subset=['year', 'CO2_emission'])  # Drop rows with missing year or CO2 values
    df['CO2_emission'] = pd.to_numeric(df['CO2_emission'], errors='coerce')
    
    st.write("Preview of the dataset:")
    st.write(df.head())

    # Time Series Decomposition
    st.header("Time Series Decomposition")

    # Set index to 'year'
    df_time_series = df[['year', 'CO2_emission']].set_index('year')

    # Decomposition
    st.subheader("Decomposed Components")

    # Check if data exists
    if not df_time_series.empty:
        decomposition = seasonal_decompose(df_time_series, model="multiplicative", period=12)
        
        # Plot the decomposition
        fig, axes = plt.subplots(4, 1, figsize=(10, 8))
        axes[0].plot(decomposition.observed, label='Observed')
        axes[0].legend(loc='upper left')
        axes[1].plot(decomposition.trend, label='Trend')
        axes[1].legend(loc='upper left')
        axes[2].plot(decomposition.seasonal, label='Seasonal')
        axes[2].legend(loc='upper left')
        axes[3].plot(decomposition.resid, label='Residual')
        axes[3].legend(loc='upper left')

        plt.tight_layout()
        st.pyplot(fig)

        # Observations
        st.write("### Observations")
        st.write("- **Observed:** The actual time series values.")
        st.write("- **Trend:** The underlying trend in the data.")
        st.write("- **Seasonal:** Recurring patterns over time.")
        st.write("- **Residual:** Noise or unexplained variations.")

    else:
        st.error("Time series data is empty. Ensure your dataset contains 'year' and 'CO2_emission' columns.")

# Footer
st.markdown("---")
st.write("This app analyzes CO2 emissions trends using time series decomposition.")
