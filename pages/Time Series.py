import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Title
st.title("CO2 Emission Time Series Analysis")

# Upload Dataset
st.header("Upload CO2 Dataset")

# Upload and process data
uploaded_file = st.file_uploader("Upload your time series file", type=["csv", "parquet"])
if uploaded_file:
    # Check file extension and read accordingly
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.parquet'):
        data = pd.read_parquet(uploaded_file)

    # Display data preview
    st.write("Data Preview:", data.head())
    st.write(f"Total Rows: {len(data)}")

# Ensure the dataset contains the required columns
if 'year' in data.columns and 'CO2_emission' in data.columns:
    # Set index to 'year'
    df_time_series = data[['year', 'CO2_emission']].set_index('year')

    # Check for missing values
    st.write("Checking for missing values...")
    st.write(df_time_series.isnull().sum())

    # Handle missing values
    df_time_series = df_time_series.dropna()  # Option 1: Drop rows with missing values
    # df_time_series = df_time_series.fillna(method='ffill')  # Option 2: Forward fill
    # df_time_series = df_time_series.interpolate(method='linear')  # Option 3: Interpolation

    # Time Series Decomposition
    st.header("Time Series Decomposition")

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
        st.error("Time series data is empty after handling missing values.")
else:
    st.error("Dataset must contain 'year' and 'CO2_emission' columns.")
