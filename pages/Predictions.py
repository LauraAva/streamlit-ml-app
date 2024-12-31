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

# Load model, encoder, and scaler from session state
model = st.session_state.get('model', None)
encoder = st.session_state.get('encoder', None)
scaler = st.session_state.get('scaler', None)

if not model:
    st.warning("Please train a model first in the 'Model Training' section.")
else:
    # File upload widget for new data
    uploaded_pred_file = st.file_uploader("Upload a new CSV file for predictions", type=["csv"])
    if uploaded_pred_file:
        new_data = pd.read_csv(uploaded_pred_file)
        st.write("### Uploaded Data Preview:")
        st.dataframe(new_data.head())

        # Ensure encoder and scaler exist
        if encoder is None or scaler is None:
            st.error("Encoder or scaler not found. Ensure these are saved during model training.")
        else:
            # Preprocess new data
            categorical_cols = ['brand', 'Model_file', 'range', 'Group', 'Country']
            missing_cols = [col for col in categorical_cols if col not in new_data.columns]
            if missing_cols:
                st.error(f"The following required columns are missing from the new dataset: {missing_cols}")
            else:
                try:
                     # Align categorical columns with encoder's expected feature set
                    st.write("Aligning categorical columns...")
                    for col in categorical_cols:
                        if col not in new_data.columns:
                            new_data[col] = "Unknown"

                    # Encode categorical features
                    st.write("Encoding categorical features...")
                    encoded_data = encoder.transform(new_data[categorical_cols])
                    encoded_df = pd.DataFrame(
                        encoded_data,
                        columns=encoder.get_feature_names_out(categorical_cols),
                        index=new_data.index
                    )

                    # Drop original categorical columns and concatenate encoded features
                    new_data = new_data.drop(columns=categorical_cols, errors="ignore")
                    new_data = pd.concat([new_data, encoded_df], axis=1)

                    # Ensure numeric data and handle NaN values
                    new_data = new_data.apply(pd.to_numeric, errors='coerce')
                    new_data = new_data.fillna(0)

                    # Align with scaler's expected feature set
                    st.write("Aligning with scaler's feature set...")
                    if len(new_data.columns) != scaler.n_features_in_:
                        st.error("Feature mismatch: Ensure the uploaded dataset has the same features as the training dataset.")
                        st.stop()

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
