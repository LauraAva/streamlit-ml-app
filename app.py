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
    
#Step 2 &3
st.header("Step 2: Data Cleaning and Integrity Check")

# 1. Full Dataset Cleaning Before Splitting
st.write("### Initial Missing Values Check:")
st.write(df[['hc', 'nox', 'hcnox']].isnull().sum())

def fill_missing_values(df, col_hc, col_nox, col_hcnox):
    # Iteratively fill missing values
    for _ in range(5):
        mask_hcnox = df[col_hcnox].isna() & df[col_hc].notna() & df[col_nox].notna()
        df.loc[mask_hcnox, col_hcnox] = df.loc[mask_hcnox, col_hc] + df.loc[mask_hcnox, col_nox]

        mask_hc = df[col_hc].isna() & df[col_nox].notna() & df[col_hcnox].notna()
        df.loc[mask_hc, col_hc] = df.loc[mask_hc, col_hcnox] - df.loc[mask_hc, col_nox]

        mask_nox = df[col_nox].isna() & df[col_hc].notna() & df[col_hcnox].notna()
        df.loc[mask_nox, col_nox] = df.loc[mask_nox, col_hcnox] - df.loc[mask_nox, col_hc]

# Apply cleaning logic
fill_missing_values(df, 'hc', 'nox', 'hcnox')

# Final cleaning for safety
df['hc'] = df['hc'].fillna(df['hc'].mean())
df['nox'] = df['nox'].fillna(df['nox'].mean())
df['hcnox'] = df['hcnox'].fillna(df['hcnox'].mean())

st.write("### Missing Values After Cleaning (Pre-Split):")
st.write(df[['hc', 'nox', 'hcnox']].isnull().sum())

# 2. Split Dataset Safely with Copies
st.write("### Splitting Data")
train_set, test_set = train_test_split(df.copy(), test_size=0.2, random_state=42)


# Final Safety Net: Refill Missing Values After Splitting
mean_hc = train_set['hc'].mean()
mean_nox = train_set['nox'].mean()
mean_hcnox = train_set['hcnox'].mean()

train_set['hc'].fillna(mean_hc, inplace=True)
train_set['nox'].fillna(mean_nox, inplace=True)
train_set['hcnox'].fillna(mean_hcnox, inplace=True)

test_set['hc'].fillna(mean_hc, inplace=True)
test_set['nox'].fillna(mean_nox, inplace=True)
test_set['hcnox'].fillna(mean_hcnox, inplace=True)

st.write("### Final Missing Values in Train Set:")
st.write(train_set[['hc', 'nox', 'hcnox']].isnull().sum())

# Save the processed train and test sets for further analysis
train_set.to_csv("final_train_set.csv", index=False)
test_set.to_csv("final_test_set.csv", index=False)

st.write("Train and Test sets saved as `final_train_set.csv` and `final_test_set.csv` for validation.")


# Clean 'Particles' Column with Mean
mean_particles_train = train_set['Particles'].mean()
train_set['Particles'].fillna(mean_particles_train, inplace=True)
test_set['Particles'].fillna(mean_particles_train, inplace=True)

# Verify Missing Values
st.write("### Missing Values in Train Set After Cleaning:")
st.write(train_set.isnull().sum())

st.write("### Missing Values in Test Set After Cleaning:")
st.write(test_set.isnull().sum())
st.write("Final Missing Values in Train Set:")
st.write(train_set.isnull().sum())

st.write("Final Missing Values in Test Set:")
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

# Step 5: Encoding Categorical Variables
st.header("Step 4: Encoding Categorical Variables")
categorical_columns = ['Carrosserie', 'fuel_type', 'Gearbox']

# Debug Step: Check for NaNs before encoding
st.write("### Checking for Missing Values Before Encoding:")
st.write(train_set[categorical_columns].isnull().sum())

encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoded_train = pd.DataFrame(encoder.fit_transform(train_set[categorical_columns]))
encoded_test = pd.DataFrame(encoder.transform(test_set[categorical_columns]))

# Concatenate Encoded Columns
train_set = pd.concat([train_set.reset_index(drop=True), encoded_train], axis=1).drop(columns=categorical_columns)
test_set = pd.concat([test_set.reset_index(drop=True), encoded_test], axis=1).drop(columns=categorical_columns)

# Debug Step: Check for alignment after encoding
train_set, test_set = train_set.align(test_set, join="left", axis=1, fill_value=0)
st.write("### Train Set and Test Set Shapes After Encoding:")
st.write(f"Train Set Shape: {train_set.shape}")
st.write(f"Test Set Shape: {test_set.shape}")



# Step 7: Model Training
st.header("Step 6: Model Training")
target_column = "CO2_class"

# Splitting features and target
X_train = train_set.drop(columns=[target_column])
y_train = train_set[target_column]
X_test = test_set.drop(columns=[target_column])
y_test = test_set[target_column]

# Debug Step: Check for NaNs in X_train and X_test
st.write("### Checking for NaN in Features Before Model Training:")
st.write(f"NaN in X_train: {X_train.isnull().sum().sum()}")
st.write(f"NaN in X_test: {X_test.isnull().sum().sum()}")

# Fill NaN if necessary
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Model Selection
model_choice = st.selectbox("Select a Model to Train", ["Logistic Regression", "Random Forest", "Decision Tree"])

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = DecisionTreeClassifier(max_depth=10, random_state=42)

# Model Training
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Debug Step: Model Training Completed
st.write("### Model Training Completed Successfully!")

# Evaluation
st.write("### Model Performance:")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))


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
