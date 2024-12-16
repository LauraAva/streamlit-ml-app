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
    # Load CSV file
    df = pd.read_csv(url, sep=',', on_bad_lines='skip', engine='python')
    st.success("Dataset loaded successfully from GitHub!")
    st.write("### Raw Dataset Preview:")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Error loading the dataset: {e}")
    st.stop()

# Step 2: Data Cleaning
st.header("Step 2: Data Cleaning")

# Fill Missing Values for Emissions
def fill_missing_values(df, col_hc, col_nox, col_hcnox):
    for _ in range(5):  # Iterate to handle dependencies
        mask_hcnox = df[col_hcnox].isna() & df[col_hc].notna() & df[col_nox].notna()
        df.loc[mask_hcnox, col_hcnox] = df.loc[mask_hcnox, col_hc] + df.loc[mask_hcnox, col_nox]

        mask_hc = df[col_hc].isna() & df[col_nox].notna() & df[col_hcnox].notna()
        df.loc[mask_hc, col_hc] = df.loc[mask_hc, col_hcnox] - df.loc[mask_hc, col_nox]

        mask_nox = df[col_nox].isna() & df[col_hc].notna() & df[col_hcnox].notna()
        df.loc[mask_nox, col_nox] = df.loc[mask_nox, col_hcnox] - df.loc[mask_nox, col_hc]

fill_missing_values(df, 'hc', 'nox', 'hcnox')

# Fill Remaining Missing Values with Mean
for col in ['hc', 'nox', 'hcnox']:
    df[col] = df[col].fillna(df[col].mean())

st.write("### Missing Values After Cleaning:")
st.write(df[['hc', 'nox', 'hcnox']].isnull().sum())

# Step 3: Train-Test Split
st.header("Step 3: Train-Test Split")

target_column = "CO2_class"  # Define the target column
if target_column not in df.columns:
    st.error(f"Target column '{target_column}' not found in the dataset.")
    st.stop()

# Split Features and Target
X = df.drop(columns=[target_column])
y = df[target_column]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write(f"Training Set Size: {X_train.shape[0]}")
st.write(f"Test Set Size: {X_test.shape[0]}")

# Step 4: Encoding Categorical Variables
st.header("Step 4: Encoding Categorical Variables")

# Define Categorical Columns
categorical_columns = ['Carrosserie', 'fuel_type', 'Gearbox']
missing_columns = [col for col in categorical_columns if col not in X_train.columns]
if missing_columns:
    st.error(f"The following columns are missing: {missing_columns}")
    st.stop()

# Fill Missing Categorical Values
for col in categorical_columns:
    X_train[col] = X_train[col].fillna('Unknown')
    X_test[col] = X_test[col].fillna('Unknown')

# Encode Categorical Variables
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # Updated argument
X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
X_test_encoded = encoder.transform(X_test[categorical_columns])

# Convert Encoded Data to DataFrame
X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_columns), index=X_train.index)
X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_columns), index=X_test.index)

# Combine Encoded Data with Numeric Columns
X_train = pd.concat([X_train.drop(columns=categorical_columns), X_train_encoded], axis=1)
X_test = pd.concat([X_test.drop(columns=categorical_columns), X_test_encoded], axis=1)


st.write("### Train Set After Encoding:")
st.dataframe(X_train.head())

# Step 5: Scaling Numerical Features
st.header("Step 5: Scaling Numerical Features")

# Scale Numerical Features
numerical_columns = ['Consumption_mix(l/100km)', 'CO2', 'power_maximal (kW)', 'Empty_mass_min(kg)']
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

st.write("### Train Set After Scaling:")
st.dataframe(X_train.head())

# Step 6: Model Training
# Drop non-numeric columns from X_train and X_test
non_numeric_columns = ['Model_UTAC', 'Commercial_name', 
                       'Code_National_Identification_Type', 'Type_Variante_Version(TVV)', 
                       'Champ_V9']
X_train = X_train.drop(columns=non_numeric_columns)
X_test = X_test.drop(columns=non_numeric_columns)

# Define necessary columns
required_columns = ['brand', 'Model_file', 'range', 'Group', 'Country']

# Check for missing columns in X_train
missing_required_columns = [col for col in required_columns if col not in X_train.columns]
if missing_required_columns:
    st.error(f"The following required columns are missing: {missing_required_columns}")
    st.stop()

# Fill missing values in required columns
for col in required_columns:
    X_train[col] = X_train[col].fillna("Unknown")
    X_test[col] = X_test[col].fillna("Unknown")

# Encode necessary categorical columns
categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_encoded = categorical_encoder.fit_transform(X_train[required_columns])
X_test_encoded = categorical_encoder.transform(X_test[required_columns])

# Convert to DataFrame
encoded_train_df = pd.DataFrame(X_train_encoded, columns=categorical_encoder.get_feature_names_out(required_columns), index=X_train.index)
encoded_test_df = pd.DataFrame(X_test_encoded, columns=categorical_encoder.get_feature_names_out(required_columns), index=X_test.index)

# Combine with the rest of the dataset
X_train = pd.concat([X_train.drop(columns=required_columns), encoded_train_df], axis=1)
X_test = pd.concat([X_test.drop(columns=required_columns), encoded_test_df], axis=1)

# Verify results
st.write("### Train Set After Encoding Required Columns:")
st.dataframe(X_train.head())
st.write("### Test Set After Encoding Required Columns:")
st.dataframe(X_test.head())

# Fill any remaining NaN values
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Verify dataset integrity
st.write(f"Remaining NaN in X_train: {X_train.isnull().sum().sum()}")
if X_train.isnull().any().any():
    st.error("NaN values still present in `X_train`. Please debug.")
    st.stop()

# Scale numerical columns
numerical_columns = ['Consumption_mix(l/100km)', 'CO2', 'power_maximal (kW)', 'Empty_mass_min(kg)']
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Verify final dataset
st.write("### Final Train Set After Scaling:")
st.dataframe(X_train.head())

st.write("### Encoded Features Preview:")
st.dataframe(X_train_encoded.iloc[:, :10].head())


st.header("Step 6: Model Training")
# Model Selection
# Map target labels
label_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
y_train = y_train.map(label_mapping)
y_test = y_test.map(label_mapping)

# Model Selection
model_choice = st.selectbox("Select a Model", ["Logistic Regression", "Random Forest", "Decision Tree"])
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = DecisionTreeClassifier(max_depth=10, random_state=42)

if model_choice == "Random Forest":
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    st.bar_chart(feature_importances.sort_values(ascending=False).head(10))

# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



# Evaluate the model
st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax)
st.pyplot(fig)

if st.checkbox("Perform Hyperparameter Tuning"):
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    st.write("Best Parameters:", grid_search.best_params_)
    model = grid_search.best_estimator_
    
if st.checkbox("Perform Feature Selection"):
    corr_matrix = X_train.corr()
    st.write("Correlation Matrix:")
    st.dataframe(corr_matrix)
    top_features = corr_matrix['target_column'].abs().sort_values(ascending=False).head(10).index.tolist()
    st.write("Top Features Based on Correlation:", top_features)

# Step 7: Model Evaluation
st.header("Step 7: Model Evaluation")

# Display Metrics
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.write("### Confusion Matrix")
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax)
st.pyplot(fig)
st.write("### Target Class Distribution in Training Set:")
st.bar_chart(y_train.value_counts())

# Step 8: Download Processed Data
st.header("Step 8: Download Processed Data")

# Save Processed Data
X_train.to_csv("processed_train_set.csv", index=False)
X_test.to_csv("processed_test_set.csv", index=False)

with open("processed_train_set.csv", "rb") as file:
    st.download_button("Download Train Set", file, "processed_train_set.csv")

with open("processed_test_set.csv", "rb") as file:
    st.download_button("Download Test Set", file, "processed_test_set.csv")
