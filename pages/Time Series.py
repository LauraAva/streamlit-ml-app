import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

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
    st.write("Data Preview:")
    st.write(data.head())
    st.write(f"Total Rows: {len(data)}")

    # Ensure the dataset contains the required columns
    if 'year' in data.columns and 'CO2_emission' in data.columns:
        # Convert 'year' to datetime
        if not pd.api.types.is_datetime64_any_dtype(data['year']):
            data['year'] = pd.to_datetime(data['year'], errors='coerce')

        # Drop rows with invalid or missing 'year'
        data = data.dropna(subset=['year'])

        # Set 'year' as the index
        df_time_series = data[['year', 'CO2_emission']].set_index('year')

        # Display summary statistics
        st.write("Summary Statistics:")
        st.write(df_time_series.describe())

        # Check for irregular frequency
        st.write("Checking index frequency...")
        freq_counts = df_time_series.index.to_series().diff().value_counts()
        st.write(freq_counts)

        # Resample data if frequency is irregular
        if freq_counts.size > 1:
            st.write("Irregular frequency detected. Resampling to monthly frequency.")
            df_time_series = df_time_series.resample('M').sum()

        # Handle missing values by interpolation
        st.write("Interpolating missing values...")
        df_time_series = df_time_series.interpolate(method='linear')

        # Log-transform the CO2_emission column to handle outliers
        st.write("Applying log transformation to CO2_emission...")
        df_time_series['CO2_emission'] = np.log1p(df_time_series['CO2_emission'])

        # Time Series Decomposition
        st.header("Time Series Decomposition")
        try:
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
        except Exception as e:
            st.error(f"Decomposition failed: {e}")
    else:
        st.error("Dataset must contain 'year' and 'CO2_emission' columns.")
else:
    st.write("Please upload a dataset to begin analysis.")

# Footer
st.markdown("---")
st.write("This app analyzes CO2 emissions trends using time series decomposition.")
