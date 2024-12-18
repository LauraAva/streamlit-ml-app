import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


st.set_page_config(page_title="Modeling", page_icon="📊")
st.title("Model Training and Evaluation")

# Check if dataset exists in session state
if 'data' not in st.session_state or st.session_state['data'] is None:
    st.error("No dataset found! Please upload or load a dataset from the 'Data Loading' page first.")
    st.stop()


# Retrieve dataset
df = st.session_state.get('data', None)
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
model_choice = st.selectbox("Choose a model:", ["Random Forest", "Logistic Regression", "Decision Tree"])
    
# Initialize the model
if model_choice == "Random Forest":
    st.write("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    if st.checkbox("Enable Hyperparameter Tuning"):
        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]}
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        st.write(f"Best Parameters: {grid_search.best_params_}")
        
    elif model_choice == "Decision Tree":
        st.write("Training Decision Tree...")
        model = DecisionTreeClassifier(random_state=42)
        if st.checkbox("Enable Hyperparameter Tuning"):
            param_grid = {"max_depth": [5, 10, 20, None], "min_samples_split": [2, 5, 10]}
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            st.write(f"Best Parameters: {grid_search.best_params_}")
    else:
        st.write("Training Logistic Regression...")
        model = LogisticRegression(max_iter=1000)

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display metrics
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Feature importance for Random Forest and Decision Tree
    if model_choice in ["Random Forest", "Decision Tree"]:
        st.write("### Feature Importance")
        feature_importances = model.feature_importances_
        sorted_indices = feature_importances.argsort()[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importances)), feature_importances[sorted_indices])
        plt.xticks(range(len(feature_importances)), X.columns[sorted_indices], rotation=90)
        plt.title("Feature Importance")
        st.pyplot(plt.gcf())

    # Save the model to session state
    st.session_state['model'] = model
