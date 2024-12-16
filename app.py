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

# Step 4: Encoding Categorical Variables
st.header("Step 4: Encoding Categorical Variables")
categorical_columns = ['Carrosserie', 'fuel_type', 'Gearbox']
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoded_train = pd.DataFrame(encoder.fit_transform(train_set[categorical_columns]))
encoded_test = pd.DataFrame(encoder.transform(test_set[categorical_columns]))

# Concatenate Encoded Columns
train_set = pd.concat([train_set.reset_index(drop=True), encoded_train], axis=1).drop(columns=categorical_columns)
test_set = pd.concat([test_set.reset_index(drop=True), encoded_test], axis=1).drop(columns=categorical_columns)

# Step 5: Scaling Numerical Features
st.header("Step 5: Scaling Numerical Features")
numerical_columns = ['Consumption_mix(l/100km)', 'CO2', 'power_maximal (kW)', 'Empty_mass_min(kg)']
scaler = StandardScaler()
train_set[numerical_columns] = scaler.fit_transform(train_set[numerical_columns])
test_set[numerical_columns] = scaler.transform(test_set[numerical_columns])

st.write("### Final Train Dataset Preview:")
st.dataframe(train_set.head())

# Step 6: Model Training
st.header("Step 6: Model Training")
target_column = "CO2_class"
X_train = train_set.drop(columns=[target_column])
y_train = train_set[target_column]
X_test = test_set.drop(columns=[target_column])
y_test = test_set[target_column]

model_choice = st.selectbox("Select a Model to Train", ["Logistic Regression", "Random Forest", "Decision Tree"])

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = DecisionTreeClassifier(max_depth=10, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write("### Model Performance:")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

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
