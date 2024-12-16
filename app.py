import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

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

# Step 1: Load Dataset from GitHub
st.header("Step 1: Load Dataset from GitHub")
url = "https://raw.githubusercontent.com/LauraAva/streamlit-ml-app/refs/heads/main/cl_union_cleaned_BI.csv"  # Replace with your URL
try:
    # Load CSV file with proper delimiter handling
    df = pd.read_csv(url, sep=',', on_bad_lines='skip', engine='python')
    st.success("Dataset loaded successfully from GitHub!")
    st.write("### Raw Dataset Preview:")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Error loading the dataset: {e}")
    st.stop()

# Step 2: Data Cleaning
st.header("Step 2: Data Cleaning")
st.write("### Filling Missing Values")
df['hcnox'] = df['hc'] + df['nox']
df['hc'] = df['hc'].fillna(df['hcnox'] - df['nox'])
df['nox'] = df['nox'].fillna(df['hcnox'] - df['hc'])
df['Particles'] = df['Particles'].fillna(df['Particles'].mean())

st.write("### Cleaned Dataset Preview:")
st.dataframe(df.head())

# Step 3: Splitting Data
st.header("Step 3: Splitting Data")
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
st.write(f"Train Set Size: {train_set.shape[0]} rows")
st.write(f"Test Set Size: {test_set.shape[0]} rows")

# Debug: Verify split integrity
st.write("### Checking for NaN values in Train/Test Split:")
st.write("Train Set Missing Values:")
st.write(train_set.isnull().sum())

st.write("Test Set Missing Values:")
st.write(test_set.isnull().sum())


# Step 4: Encoding Categorical Variables
st.header("Step 4: Encoding Categorical Variables")

# Select categorical columns
categorical_columns = ['Carrosserie', 'fuel_type', 'Gearbox']

# Check for missing values in categorical columns
for col in categorical_columns:
    if df[col].isnull().any():
        st.warning(f"Missing values detected in {col}. Filling with 'Unknown'.")
        df[col] = df[col].fillna('Unknown')

try:
    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")  # For newer scikit-learn
    encoded_train = encoder.fit_transform(train_set[categorical_columns])
    encoded_test = encoder.transform(test_set[categorical_columns])

    # Convert to DataFrame with column names
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_columns))
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_columns))

    # Reset indices and concatenate encoded data
    train_set = pd.concat([train_set.reset_index(drop=True), encoded_train_df], axis=1).drop(columns=categorical_columns)
    test_set = pd.concat([test_set.reset_index(drop=True), encoded_test_df], axis=1).drop(columns=categorical_columns)

    st.success("Categorical encoding completed successfully!")
    st.write("### Train Set After Encoding:")
    st.dataframe(train_set.head())

except Exception as e:
    st.error(f"Error during OneHotEncoding: {e}")
    st.stop()

# Step 5: Scaling Numerical Features
st.header("Step 5: Scaling Numerical Features")
numerical_columns = ['Consumption_mix(l/100km)', 'CO2', 'power_maximal (kW)', 'Empty_mass_min(kg)']
scaler = StandardScaler()
train_set[numerical_columns] = scaler.fit_transform(train_set[numerical_columns])
test_set[numerical_columns] = scaler.transform(test_set[numerical_columns])

st.write("### Final Train Dataset Preview:")
st.dataframe(train_set.head())

# Step 6: Verify Data Integrity Before Model Training
st.header("Step 6: Verify Data Integrity Before Model Training")

# Check for NaN in train and test datasets
st.write("Checking for NaN values in Train Set:")
st.write(X_train.isnull().sum())

st.write("Checking for NaN values in Test Set:")
st.write(X_test.isnull().sum())

# Fill NaN values if found
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())
st.write("NaN values filled with column means.")


# Step 6: Model Training
st.header("Step 6: Model Training")

# Define Target and Features
target_column = "CO2_class"
if target_column not in train_set.columns:
    st.error(f"Target column '{target_column}' not found in the dataset.")
    st.stop()

# Drop target column and ensure all remaining features are numeric
X_train = train_set.drop(columns=[target_column])
y_train = train_set[target_column]
X_test = test_set.drop(columns=[target_column])
y_test = test_set[target_column]

# Check for NaN and non-numeric columns
if X_train.isnull().any().any() or X_test.isnull().any().any():
    st.warning("NaN values detected in features. Filling NaN with column mean.")
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())

# Ensure all columns are numeric
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# Model Selection
st.write("### Choose Model for Training")
model_choice = st.selectbox("Select a Model", ["Logistic Regression", "Random Forest", "Decision Tree"])

# Initialize Model
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = DecisionTreeClassifier(max_depth=10, random_state=42)

try:
    # Train the Model
    model.fit(X_train, y_train)

    # Make Predictions
    y_pred = model.predict(X_test)

    # Model Performance Metrics
    st.write("### Model Performance Metrics")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.header("Step 7: Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax)
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error during model training or prediction: {e}")
    st.stop()


# Step 7: Visualization
st.header("Step 7: Confusion Matrix")
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
st.pyplot(plt.gcf())

# Step 8: Download Processed Data
st.header("Step 8: Download Processed Data")
train_set_cleaned = train_set
test_set_cleaned = test_set

# Download Buttons
st.write("Download Train and Test Datasets")
train_set_cleaned.to_csv("train_set_cleaned.csv", index=False)
test_set_cleaned.to_csv("test_set_cleaned.csv", index=False)

with open("train_set_cleaned.csv", "rb") as file:
    st.download_button("Download Train Set", file, "train_set_cleaned.csv")

with open("test_set_cleaned.csv", "rb") as file:
    st.download_button("Download Test Set", file, "test_set_cleaned.csv")
