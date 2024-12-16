import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
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

# Display Class Distribution
st.write("### Class Distribution in Target Variable (y):")
st.write("Training Set Distribution:")
st.bar_chart(y.value_counts())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
st.write(f"Training Set Size: {X_train.shape[0]}")
st.write(f"Test Set Size: {X_test.shape[0]}")

# Step 4: Encoding Categorical Variables
st.header("Step 4: Encoding Categorical Variables")
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
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
X_test_encoded = encoder.transform(X_test[categorical_columns])

# Convert Encoded Data to DataFrame
X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_columns), index=X_train.index)
X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_columns), index=X_test.index)

# Combine Encoded Data with Numeric Columns
X_train = pd.concat([X_train.drop(columns=categorical_columns), X_train_encoded], axis=1)
X_test = pd.concat([X_test.drop(columns=categorical_columns), X_test_encoded], axis=1)

# Preview Encoded Features
st.write("### Preview of Encoded Features:")
st.dataframe(X_train_encoded.head())

# Step 5: Scaling Numerical Features
st.header("Step 5: Scaling Numerical Features")
numerical_columns = ['Consumption_mix(l/100km)', 'CO2', 'power_maximal (kW)', 'Empty_mass_min(kg)']
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Feature Selection Based on Correlation
if st.checkbox("Perform Feature Selection"):
    corr_matrix = X_train.corr()
    st.write("Correlation Matrix:")
    st.dataframe(corr_matrix)
    top_features = corr_matrix[target_column].abs().sort_values(ascending=False).head(10).index.tolist()
    st.write("Top Features Based on Correlation:", top_features)

# Step 6: Model Training and Hyperparameter Tuning
st.header("Step 6: Model Training")
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


st.header("Step 7: Model Training")
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

if model_choice == "Random Forest":
    st.write("### Feature Importance")
    feature_importances = model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(X_train.shape[1]), feature_importances[sorted_indices])
    plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90)
    plt.title("Feature Importance for Random Forest")
    st.pyplot(plt.gcf())
    
st.write("### Class Distribution in Training and Test Sets")
st.write("y_train distribution:")
st.bar_chart(y_train.value_counts())
st.write("y_test distribution:")
st.bar_chart(y_test.value_counts())

# Check for imbalance
if y_train.value_counts().max() / y_train.value_counts().min() > 1.5:
    st.warning("Class imbalance detected in y_train. Consider oversampling or undersampling.")


st.write("### Encoded Features Preview")
st.write("Top Encoded Features in Train Set:")
st.write(X_train.columns[:10])

if st.checkbox("Enable Hyperparameter Tuning"):
    st.write("### Hyperparameter Tuning")
    if model_choice == "Random Forest":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        st.write(f"Best Parameters: {grid_search.best_params_}")
        st.write(f"Best Accuracy: {grid_search.best_score_:.2f}")
        model = grid_search.best_estimator_

        # Re-train and evaluate the best model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write(f"Model Accuracy After Tuning: {accuracy_score(y_test, y_pred):.2f}")

if st.checkbox("Enable Hyperparameter Tuning"):
    st.write("### Hyperparameter Tuning")
    if model_choice == "Random Forest":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        st.write(f"Best Parameters: {grid_search.best_params_}")
        st.write(f"Best Accuracy: {grid_search.best_score_:.2f}")
        model = grid_search.best_estimator_

        # Re-train with the tuned model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write(f"Model Accuracy After Tuning: {accuracy_score(y_test, y_pred):.2f}")

# Apply feature selection if checkbox is enabled
if st.checkbox("Select Top Features After Tuning"):
    st.write("### Selecting Top Features After Tuning")
    correlation = X_train.corrwith(y_train)
    top_features = correlation.abs().sort_values(ascending=False).head(10).index
    st.write(f"Top Features Based on Correlation: {list(top_features)}")

    # Reduce feature set
    X_train = X_train[top_features]
    X_test = X_test[top_features]

    # Re-train the model with the reduced feature set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model again
    st.write(f"Model Accuracy After Feature Selection: {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report After Feature Selection:")
    st.text(classification_report(y_test, y_pred))


# Step 8: Download Processed Data
st.header("Step 8: Download Processed Data")
X_train.to_csv("processed_train_set.csv", index=False)
X_test.to_csv("processed_test_set.csv", index=False)

with open("processed_train_set.csv", "rb") as file:
    st.download_button("Download Train Set", file, "processed_train_set.csv")

with open("processed_test_set.csv", "rb") as file:
    st.download_button("Download Test Set", file, "processed_test_set.csv")
