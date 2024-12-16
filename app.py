import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

# App title and introduction
st.title("Machine Learning Pipeline for CO2 Emissions Analysis")
st.markdown("""
### Overview
This application provides a complete machine learning pipeline for analyzing CO2 emissions. Features include:
1. Data preprocessing
2. Feature encoding
3. Model training
4. Evaluation and visualization
""")

# Load the dataset directly
dataset_path = "cl_union_cleaned_BI.csv"  # File in the root directory
try:
    data = pd.read_csv(dataset_path)
    st.write("### Dataset Overview:")
    st.write(data.head())

except FileNotFoundError:
    st.error("Dataset not found! Ensure the file is uploaded to the same directory as your app.")
    st.stop()  

    # Dataset information
    st.write("### Dataset Information")
    buffer = []
    data.info(buf=buffer)
    st.text("\n".join(buffer))

    # Handling missing values
    st.sidebar.header("Step 2: Handle Missing Values")
    fill_missing_option = st.sidebar.selectbox("Select a method for handling missing values:", ["Fill with Mean", "Fill with Median", "Drop Rows with Missing Values"])

    if fill_missing_option == "Fill with Mean":
        data = data.fillna(data.mean())
    elif fill_missing_option == "Fill with Median":
        data = data.fillna(data.median())
    elif fill_missing_option == "Drop Rows with Missing Values":
        data = data.dropna()

    st.write("### Cleaned Dataset Preview")
    st.dataframe(data.head())

    # Splitting data
    st.sidebar.header("Step 3: Split Data")
    target_column = st.sidebar.selectbox("Select the Target Variable", data.columns)
    if target_column:
        X = data.drop(columns=[target_column])
        y = data[target_column]

        test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

        st.write("### Data Split Overview")
        st.write(f"Training Set: {X_train.shape[0]} rows")
        st.write(f"Test Set: {X_test.shape[0]} rows")

    # Feature encoding
    st.sidebar.header("Step 4: Encode Categorical Features")
    categorical_columns = st.sidebar.multiselect("Select Categorical Columns", X.columns)
    if categorical_columns:
        encoder = OneHotEncoder(sparse=False)
        X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
        X_test_encoded = encoder.transform(X_test[categorical_columns])

        encoded_columns = encoder.get_feature_names_out(categorical_columns)
        X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
        X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)

        X_train = X_train.drop(columns=categorical_columns).join(X_train_encoded_df)
        X_test = X_test.drop(columns=categorical_columns).join(X_test_encoded_df)

        st.write("### Feature Encoding Completed")
        st.write("Training Set Shape:", X_train.shape)

    # Standardizing data
    st.sidebar.header("Step 5: Standardize Features")
    numeric_columns = st.sidebar.multiselect("Select Numeric Columns", X.columns)
    if numeric_columns:
        scaler = StandardScaler()
        X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
        X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

        st.write("### Standardization Completed")

    # Model training and evaluation
    st.sidebar.header("Step 6: Train Model")
    model_choice = st.sidebar.selectbox("Select a Model", ["Random Forest", "Gradient Boosting"])

    if model_choice == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", min_value=50, max_value=500, value=100, step=50)
        max_depth = st.sidebar.slider("Max Depth", min_value=5, max_value=50, value=20, step=5)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_choice == "Gradient Boosting":
        learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
        n_estimators = st.sidebar.slider("Number of Estimators", min_value=50, max_value=500, value=100, step=50)
        model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model evaluation
    st.write("### Model Performance")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.write("### Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    st.pyplot(plt.gcf())

    # Feature importance (for tree-based models)
    if hasattr(model, "feature_importances_"):
        st.write("### Feature Importance")
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(importance_df.set_index("Feature"))

    # SMOTE for imbalance handling
    st.sidebar.header("Step 7: Handle Imbalanced Data")
    use_smote = st.sidebar.checkbox("Apply SMOTE")
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        st.write("Class Distribution After SMOTE:")
        st.write(pd.Series(y_train_resampled).value_counts())

    # Data visualization
    st.sidebar.header("Step 8: Visualizations")
    if st.sidebar.checkbox("Visualize CO2 Distribution"):
        st.write("### CO2 Emissions Distribution")
        plt.figure(figsize=(10, 6))
        plt.hist(data['CO2'], bins=30, color='blue', alpha=0.7, label='CO2')
        plt.xlabel('CO2 Emissions')
        plt.ylabel('Frequency')
        plt.title('CO2 Emissions Distribution')
        st.pyplot(plt.gcf())

else:
    st.write("Please upload a dataset to proceed.")


