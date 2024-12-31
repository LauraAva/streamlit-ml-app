import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import setup_sidebar
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.feature_selection import VarianceThreshold

# Set up the sidebar
setup_sidebar()

st.title("🧪Model Training and Evaluation")
st.write("Upload your dataset here or use the preloaded dataset.")

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['data'] = df
    st.success("Dataset loaded successfully!")
else:
    st.warning("Please upload a dataset.")


# Step 1: Retrieve dataset from session state
if 'data' not in st.session_state or st.session_state['data'] is None:
    st.error("No dataset found! Please upload or load a dataset from the 'Data Loading' page first.")
    st.stop()

df = st.session_state['data']  # Retrieve dataset

# Step 2: Verify target column
target_column = "CO2_class"
if target_column not in df.columns:
    st.error(f"Target column '{target_column}' not found in the dataset!")
    st.stop()

# Step 3: Split Features (X) and Target (y)
st.write("### Preprocessing Steps")
X = df.drop(columns=[target_column])
y = df[target_column]

# Step 4: Verify categorical columns
categorical_cols = ['brand', 'Model_file', 'range', 'Group', 'Country']
missing_cols = [col for col in categorical_cols if col not in X.columns]
if missing_cols:
    st.error(f"The following categorical columns are missing from the dataset: {missing_cols}")
    st.stop()

# Step 5: Encode categorical features
st.write("Encoding categorical features...")
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_array = encoder.fit_transform(X[categorical_cols])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))

# Step 6: Drop original categorical columns and concatenate encoded features
X = X.drop(columns=categorical_cols, errors='ignore')
X = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Step 7: Ensure numeric data and handle NaN values
X = X.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
X = X.fillna(0)  # Replace NaN values with 0

# Step 7a: Apply VarianceThreshold to remove low-variance features
# Apply VarianceThreshold to remove low-variance features
st.write("Applying Variance Threshold to remove low-variance features...")
selector = VarianceThreshold(threshold=0.01)

# Fit and transform
X_selected = selector.fit_transform(X)

# Debugging: Check shapes and selected columns
feature_variances = selector.variances_
st.write(f"Feature variances: {feature_variances}")
st.write(f"Shape of X before VarianceThreshold: {X.shape}")
st.write(f"Shape of X after VarianceThreshold: {X_selected.shape}")

selected_columns = X.columns[selector.get_support()]
st.write(f"Number of selected columns: {len(selected_columns)}")
st.write(f"Selected columns: {selected_columns}")

# Check for mismatched shapes
if X_selected.shape[1] == len(selected_columns):
    X = pd.DataFrame(X_selected, columns=selected_columns)
    st.success(f"VarianceThreshold applied successfully. {X.shape[1]} features retained.")
else:
    st.error(f"Shape mismatch: X_selected has {X_selected.shape[1]} features, but selected_columns has {len(selected_columns)} columns.")
    st.stop()


# Step 8: Scale numerical features
st.write("Scaling numerical features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

st.success("Preprocessing completed successfully!")

# Step 9: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Model selection and training
# Initialize model placeholder
model = None

# Model selection and training
st.write("### Model Training")
model_choice = st.selectbox("Choose a model:", ["Random Forest", "Logistic Regression", "Decision Tree"])

try:
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
        model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
        if st.checkbox("Enable Hyperparameter Tuning"):
            param_grid = {"max_depth": [5, 10, 20, None], "min_samples_split": [2, 5, 10]}
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            st.write(f"Best Parameters: {grid_search.best_params_}")

    elif model_choice == "Logistic Regression":
        st.write("Training Logistic Regression...")
        model = LogisticRegression(
            max_iter=500, solver='saga', tol=1e-3, n_jobs=-1, class_weight='balanced'
        )
        import time
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        st.write(f"Logistic Regression training took {end_time - start_time:.2f} seconds.")

    else:  # Correctly placed and aligned else block
        st.error("No model selected.")
        st.stop()
    
    # Train the model and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save the trained model in session state
    st.session_state['model'] = model

    # Evaluation metrics
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
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

        # Calculate cumulative importance
        cumulative_importance = feature_importances[sorted_indices].cumsum()
        
        # Show top features
        top_n = st.slider("Select top N features to display:", min_value=5, max_value=20, value=10)
        top_features = sorted_indices[:top_n]


        # Plot feature importance with cumulative percentage
        plt.figure(figsize=(12, 8))
        plt.bar(range(top_n), feature_importances[top_features], align="center", label="Feature Importance")
        plt.plot(range(top_n), cumulative_importance[:top_n], marker="o", color="red", label="Cumulative Importance")
        plt.xticks(range(top_n), X.columns[top_features], rotation=45, ha="right")
        plt.title(f"Top {top_n} Feature Importance with Cumulative Contribution")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.legend(loc="upper left")
        plt.tight_layout()
        st.pyplot(plt.gcf())
    
except Exception as e:
    st.error(f"An error occurred during model training or evaluation: {str(e)}")


st.write("### Encoded Features Summary")
encoded_summary = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))
st.write(encoded_summary.describe())

# Step 12: Save the model to session state
st.session_state['model'] = model
st.success(f"{model_choice} model training completed successfully!")
