import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Helper Functions
def preprocess_data(df, target_column):
    """Cleans and prepares the dataset."""
    # Fill missing emission values
    def fill_missing_values(df, col_hc, col_nox, col_hcnox):
        for _ in range(5):  # Iterate to handle dependencies
            mask_hcnox = df[col_hcnox].isna() & df[col_hc].notna() & df[col_nox].notna()
            df.loc[mask_hcnox, col_hcnox] = df.loc[mask_hcnox, col_hc] + df.loc[mask_hcnox, col_nox]

            mask_hc = df[col_hc].isna() & df[col_nox].notna() & df[col_hcnox].notna()
            df.loc[mask_hc, col_hc] = df.loc[mask_hc, col_hcnox] - df.loc[mask_hc, col_nox]

            mask_nox = df[col_nox].isna() & df[col_hc].notna() & df[col_hcnox].notna()
            df.loc[mask_nox, col_nox] = df.loc[mask_nox, col_hcnox] - df.loc[mask_nox, col_hc]

    fill_missing_values(df, 'hc', 'nox', 'hcnox')
    for col in ['hc', 'nox', 'hcnox']:
        df[col] = df[col].fillna(df[col].mean())

    return df

def encode_and_scale(X_train, X_test, categorical_columns, numerical_columns):
    """Encodes categorical columns and scales numerical columns."""
    # Encode Categorical Variables
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
    X_test_encoded = encoder.transform(X_test[categorical_columns])

    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_columns), index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_columns), index=X_test.index)

    X_train = pd.concat([X_train.drop(columns=categorical_columns), X_train_encoded], axis=1)
    X_test = pd.concat([X_test.drop(columns=categorical_columns), X_test_encoded], axis=1)

    # Scale Numerical Features
    scaler = StandardScaler()
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    return X_train, X_test

def visualize_feature_importance(model, feature_names):
    """Visualizes feature importance for RandomForest."""
    feature_importances = pd.Series(model.feature_importances_, index=feature_names)
    st.bar_chart(feature_importances.sort_values(ascending=False).head(10))

# Title and Introduction
st.title("CO2 Emission Analysis & Machine Learning Pipeline")
st.write("""
Upload or load a dataset to execute a complete end-to-end workflow:
1. **Data Cleaning**
2. **Feature Encoding and Scaling**
3. **Model Training**
4. **Visualization**
5. **Results Analysis**
""")

# Step 1: Load Dataset
st.header("Step 1: Load Dataset")
url = "https://raw.githubusercontent.com/LauraAva/streamlit-ml-app/refs/heads/main/cl_union_cleaned_BI.csv"  # Replace with your URL
try:
    df = pd.read_csv(url, sep=',', on_bad_lines='skip', engine='python')
    st.success("Dataset loaded successfully!")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Step 2: Preprocessing
st.header("Step 2: Data Preprocessing")
target_column = "CO2_class"
if target_column not in df.columns:
    st.error(f"Target column '{target_column}' not found.")
    st.stop()

df = preprocess_data(df, target_column)

# Train-Test Split
st.header("Step 3: Train-Test Split")
X = df.drop(columns=[target_column])
y = df[target_column]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

st.write(f"Train Size: {X_train.shape[0]}, Test Size: {X_test.shape[0]}")

# Step 4: Encoding and Scaling
st.header("Step 4: Encoding & Scaling")
categorical_columns = ['Carrosserie', 'fuel_type', 'Gearbox']
numerical_columns = ['Consumption_mix(l/100km)', 'CO2', 'power_maximal (kW)', 'Empty_mass_min(kg)']
X_train, X_test = encode_and_scale(X_train, X_test, categorical_columns, numerical_columns)
st.write("Encoding and scaling completed.")

# Step 5: Model Training
st.header("Step 5: Model Training")
label_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
y_train = y_train.map(label_mapping)
y_test = y_test.map(label_mapping)

model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "Decision Tree"])
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = DecisionTreeClassifier(max_depth=10, random_state=42)

# Train and Evaluate
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Feature Importance
if model_choice == "Random Forest":
    visualize_feature_importance(model, X_train.columns)

# Step 6: Model Tuning (Optional)
if st.checkbox("Enable Hyperparameter Tuning"):
    param_grid = {'max_depth': [5, 10, 20], 'n_estimators': [50, 100, 200]}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    st.write(f"Best Parameters: {grid_search.best_params_}")

# Step 7: Confusion Matrix
st.header("Step 7: Confusion Matrix")
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax)
st.pyplot(fig)

# Step 8: Data Download
st.header("Step 8: Download Processed Data")
X_train.to_csv("processed_train_set.csv", index=False)
X_test.to_csv("processed_test_set.csv", index=False)
with open("processed_train_set.csv", "rb") as file:
    st.download_button("Download Train Set", file, "processed_train_set.csv")
with open("processed_test_set.csv", "rb") as file:
    st.download_button("Download Test Set", file, "processed_test_set.csv")
