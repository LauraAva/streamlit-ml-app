import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression  # Add this import

st.title("Model Training and Evaluation")

# Retrieve dataset
df = st.session_state.get('data', None)
if df is not None:
    # Preprocess data
    st.write("### Preprocessing Steps")
    target_column = "CO2_class"
    if target_column not in df.columns:
        st.error("Target column 'CO2_class' not found!")
        st.stop()

    X = df.drop(columns=[target_column])
    y = df[target_column]

   # Encode categorical features
categorical_cols = ['brand', 'Model_file', 'range', 'Group', 'Country']
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Perform encoding and get feature names
encoded_array = encoder.fit_transform(X[categorical_cols])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))

# Drop original categorical columns and concatenate encoded features
X = X.drop(columns=categorical_cols)
X = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Ensure all columns are numeric and handle NaN values
X = X.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
X = X.fillna(0)  # Replace any NaN values with 0

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

st.success("Preprocessing completed successfully!")


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
st.write("### Model Training")
model_choice = st.selectbox("Choose a model:", ["Random Forest", "Logistic Regression"])
if model_choice == "Random Forest":
    model = RandomForestClassifier()
    if st.checkbox("Enable Hyperparameter Tuning"):
        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]}
        grid_search = GridSearchCV(model, param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        st.write(f"Best Parameters: {grid_search.best_params_}")
    else:
        model = LogisticRegression(max_iter=1000)

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display metrics
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Feature importance for Random Forest
    if model_choice == "Random Forest":
        st.write("### Feature Importance")
        feature_importances = model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importances)), feature_importances)
        plt.title("Feature Importance")
        st.pyplot(plt.gcf())

    # Save the model to session state
    st.session_state['model'] = model
else:
    st.warning("Please load a dataset first.")
