import streamlit as st
import pandas as pd
from utils import setup_sidebar

# Set up the sidebar
setup_sidebar()

st.title("🔮Prediction Interface")
st.write("Upload your dataset here or use the preloaded dataset.")

# File upload widget for base dataset
uploaded_file = st.file_uploader("Upload a CSV file (for reference schema)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['data'] = df
    st.success("Dataset loaded successfully!")
else:
    st.warning("Please upload a dataset.")

# Load model, scaler, and expected features from session state
model = st.session_state.get('model', None)
scaler = st.session_state.get('scaler', None)
expected_features = st.session_state.get('expected_features', None)

if not model or not scaler or not expected_features:
    st.warning("Please ensure the model, scaler, and expected features are saved during training.")
else:
    # File upload widget for new data
    uploaded_pred_file = st.file_uploader("Upload a new CSV file for predictions", type=["csv"])
    if uploaded_pred_file:
        new_data = pd.read_csv(uploaded_pred_file)
        st.write("### Uploaded Data Preview:")
        st.dataframe(new_data.head())

        try:
            # Align features with the expected schema
            st.write("Aligning features with the expected schema...")
            for feature in expected_features:
                if feature not in new_data.columns:
                    new_data[feature] = 0  # Add missing features with a default value

            # Reorder columns to match the expected feature order
            new_data = new_data[expected_features]

            # Scale the data
            st.write("Scaling numerical features...")
            new_data_scaled = scaler.transform(new_data)
            new_data = pd.DataFrame(new_data_scaled, columns=new_data.columns)

            # Make predictions
            predictions = model.predict(new_data)
            new_data['Predictions'] = predictions

            # Display predictions
            st.write("### Predictions:")
            st.dataframe(new_data)

            # Download predictions
            csv = new_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"An error occurred during preprocessing or prediction: {e}")
