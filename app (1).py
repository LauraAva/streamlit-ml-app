
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

# Title and Introduction
st.title("Machine Learning for CO2 Emissions Analysis")
st.write(
    "This application provides a complete end-to-end machine learning workflow for analyzing CO2 emissions. "
    "You can preprocess data, train models, and explore insights in a seamless interface."
)

# Upload CSV
uploaded_file = st.file_uploader("/content/drive/MyDrive/DATA ANALYST/cl_union_cleaned_today.csv", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Overview:")
    st.write(data.head())

    # Data Information
    st.write("Dataset Info:")
    st.write(str(data.info()))

    # Step 1: Preprocessing
    st.header("Step 1: Preprocessing")
    st.write("### Cleaning missing values")
    data['hc'] = data['hc'].fillna(data['hc'].mean())
    data['nox'] = data['nox'].fillna(data['nox'].mean())
    data['hcnox'] = data['hcnox'].fillna(data['hcnox'].mean())
    st.write("Missing values filled with mean for numeric columns.")

    # Display cleaned data
    st.write("Cleaned Data Preview:")
    st.write(data.head())

    # Step 2: Splitting Data
    st.header("Step 2: Splitting Data")
    target_variable = st.selectbox("Select Target Variable", options=data.columns)
    if target_variable:
        st.write(f"Target variable: {target_variable}")
        X = data.drop(columns=[target_variable])
        y = data[target_variable]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"Training set size: {X_train.shape[0]} rows")
        st.write(f"Test set size: {X_test.shape[0]} rows")

    # Step 3: Feature Encoding
    st.header("Step 3: Feature Encoding")
    categorical_columns = st.multiselect("Select categorical columns to encode", options=X.columns)
    if categorical_columns:
        encoder = OneHotEncoder()
        X_train_encoded = encoder.fit_transform(X_train[categorical_columns]).toarray()
        X_test_encoded = encoder.transform(X_test[categorical_columns]).toarray()
        st.write(f"Encoded {len(categorical_columns)} columns.")

    # Step 4: Standardizing Data
    st.header("Step 4: Standardizing Data")
    numeric_columns = st.multiselect("Select numeric columns to standardize", options=X.columns)
    if numeric_columns:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
        X_test_scaled = scaler.transform(X_test[numeric_columns])
        st.write(f"Standardized {len(numeric_columns)} columns.")

    # Step 5: Train Random Forest Classifier
    st.header("Step 5: Train Model")
    st.subheader("Random Forest Classifier")
    n_estimators = st.slider("Number of Trees", min_value=50, max_value=500, value=100, step=50)
    max_depth = st.slider("Max Depth", min_value=5, max_value=50, value=20, step=5)
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)

    # Model Evaluation
    st.write("### Model Performance")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred_rf))

    # Feature Importance
    st.write("### Feature Importance")
    feature_importances = rf_classifier.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    st.bar_chart(importance_df.set_index("Feature"))

    # Confusion Matrix
    st.write("### Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, cmap="Blues")
    st.pyplot(plt.gcf())

    # Step 6: Oversampling with SMOTE
    st.header("Step 6: Handling Imbalanced Data")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    st.write("Class distribution after SMOTE:")
    st.write(pd.Series(y_resampled).value_counts())

    # Step 7: Visualizations
    st.header("Step 7: Visualizations")
    st.write("### CO2 Distribution")
    plt.figure(figsize=(10, 6))
    plt.hist(data['CO2'], bins=30, color='blue', alpha=0.7, label='CO2')
    plt.xlabel('CO2 Emissions')
    plt.ylabel('Frequency')
    plt.title('CO2 Emissions Distribution')
    st.pyplot(plt.gcf())
else:
    st.write("Please upload a dataset to proceed.")
