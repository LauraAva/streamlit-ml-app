import streamlit as st
import pandas as pd

st.title("Prediction Interface")

# Load model from session state
model = st.session_state.get('model', None)
if model:
    uploaded_file = st.file_uploader("Upload a new CSV file for predictions", type=["csv"])
    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview:")
        st.dataframe(new_data.head())

        # Preprocess data (similar steps as before)
        categorical_cols = ['brand', 'Model_file', 'range', 'Group', 'Country']
        encoder = st.session_state.get('encoder', None)
        scaler = st.session_state.get('scaler', None)

        if encoder and scaler:
            encoded_data = encoder.transform(new_data[categorical_cols])
            new_data = pd.concat([new_data.drop(columns=categorical_cols), pd.DataFrame(encoded_data)], axis=1)
            scaled_data = scaler.transform(new_data)
            new_data = pd.DataFrame(scaled_data, columns=new_data.columns)

            # Make predictions
            predictions = model.predict(new_data)
            new_data['Predictions'] = predictions
            st.write("### Predictions:")
            st.dataframe(new_data)

            # Download predictions
            new_data.to_csv("predictions.csv", index=False)
            st.download_button("Download Predictions", "predictions.csv")
else:
    st.warning("Please train a model first in the 'Model Training' section.")
