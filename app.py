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
    
st.header("X_TRAIN NOT DEFINED")
if target_column not in train_set.columns:
    st.error(f"Target column '{target_column}' not found in the dataset.")
    st.stop()

# Ensure numeric features only
X_train = train_set.drop(columns=[target_column]).select_dtypes(include=[np.number])
y_train = train_set[target_column]
X_test = test_set.drop(columns=[target_column]).select_dtypes(include=[np.number])
y_test = test_set[target_column]

# Handle missing values
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

st.write("### Features and Target Split Completed!")
st.write(f"X_train shape: {X_train.shape}")
st.write(f"X_test shape: {X_test.shape}")


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

train_set['hc'] = train_set['hc'].fillna(mean_hc)
train_set['nox'] = train_set['nox'].fillna(mean_nox)
train_set['hcnox'] = train_set['hcnox'].fillna(mean_hcnox)

test_set['hc'] = test_set['hc'].fillna(mean_hc)
test_set['nox'] = test_set['nox'].fillna(mean_nox)
test_set['hcnox'] = test_set['hcnox'].fillna(mean_hcnox)
 

# Save the processed train and test sets for further analysis
train_set.to_csv("final_train_set.csv", index=False)
test_set.to_csv("final_test_set.csv", index=False)
# Define the function to fill missing values for consumption columns
def fill_consumption_values(df, urban_col, extra_urban_col, mix_col):
    for _ in range(5):  # Iterate to handle dependencies
        # Fill Consumption_mix if Urban and Extra-Urban are available
        mask_mix = df[mix_col].isna() & df[urban_col].notna() & df[extra_urban_col].notna()
        df.loc[mask_mix, mix_col] = (df.loc[mask_mix, urban_col] + df.loc[mask_mix, extra_urban_col]) / 2

        # Fill Urban_consumption if Extra-Urban and Mix are available
        mask_urban = df[urban_col].isna() & df[extra_urban_col].notna() & df[mix_col].notna()
        df.loc[mask_urban, urban_col] = 2 * df.loc[mask_urban, mix_col] - df.loc[mask_urban, extra_urban_col]

        # Fill Extra-Urban consumption if Urban and Mix are available
        mask_extra_urban = df[extra_urban_col].isna() & df[urban_col].notna() & df[mix_col].notna()
        df.loc[mask_extra_urban, extra_urban_col] = 2 * df.loc[mask_extra_urban, mix_col] - df.loc[mask_extra_urban, urban_col]

# Apply the function to train and test sets
fill_consumption_values(train_set, 'Urban_consumption (l/100km)', 'Extra_urban_consumption(l/100km)', 'Consumption_mix(l/100km)')
fill_consumption_values(test_set, 'Urban_consumption (l/100km)', 'Extra_urban_consumption(l/100km)', 'Consumption_mix(l/100km)')

# Fill remaining missing values in train set with column means
train_set['Urban_consumption (l/100km)'] = train_set['Urban_consumption (l/100km)'].fillna(train_set['Urban_consumption (l/100km)'].mean())
train_set['Extra_urban_consumption(l/100km)'] = train_set['Extra_urban_consumption(l/100km)'].fillna(train_set['Extra_urban_consumption(l/100km)'].mean())
train_set['Consumption_mix(l/100km)'] = train_set['Consumption_mix(l/100km)'].fillna(train_set['Consumption_mix(l/100km)'].mean())
train_set['CO_type_I (g/km)'] = train_set['CO_type_I (g/km)'].fillna(train_set['CO_type_I (g/km)'].mean())

# Fill remaining missing values in test set with column means
test_set['Urban_consumption (l/100km)'] = test_set['Urban_consumption (l/100km)'].fillna(train_set['Urban_consumption (l/100km)'].mean())  # Use train mean to avoid leakage
test_set['Extra_urban_consumption(l/100km)'] = test_set['Extra_urban_consumption(l/100km)'].fillna(train_set['Extra_urban_consumption(l/100km)'].mean())  # Use train mean
test_set['Consumption_mix(l/100km)'] = test_set['Consumption_mix(l/100km)'].fillna(train_set['Consumption_mix(l/100km)'].mean())  # Use train mean
test_set['CO_type_I (g/km)'] = test_set['CO_type_I (g/km)'].fillna(train_set['CO_type_I (g/km)'].mean())  # Use train mean

# Verify missing values in train and test sets
st.write("### Final Missing Values in Train Set (Consumption Columns):")
st.write(train_set[['Urban_consumption (l/100km)', 'Extra_urban_consumption(l/100km)', 
                    'Consumption_mix(l/100km)', 'CO_type_I (g/km)']].isnull().sum())

st.write("### Final Missing Values in Test Set (Consumption Columns):")
st.write(test_set[['Urban_consumption (l/100km)', 'Extra_urban_consumption(l/100km)', 
                   'Consumption_mix(l/100km)', 'CO_type_I (g/km)']].isnull().sum())

# Save final train and test sets for debugging
train_set.to_csv("final_train_set_consumption.csv", index=False)
test_set.to_csv("final_test_set_consumption.csv", index=False)
st.write("Train and Test sets saved as `final_train_set_consumption.csv` and `final_test_set_consumption.csv` for debugging.")


# Clean 'Particles' Column with Mean
mean_particles_train = train_set['Particles'].mean()
train_set['Particles'].fillna(mean_particles_train, inplace=True)
test_set['Particles'].fillna(mean_particles_train, inplace=True)

###Check 

st.write("### Train Set Preview:")
st.dataframe(train_set.head())

st.write("### Test Set Preview:")
st.dataframe(test_set.head())
st.write("### Train Set Preview:")
st.dataframe(train_set.head())

st.write("### Test Set Preview:")
st.dataframe(test_set.head())

######before encoding
st.header("before encoding")
# Ensure categorical columns have no NaNs or invalid data
for col in categorical_columns:
    if train_set[col].isnull().any():
        train_set[col] = train_set[col].fillna("Unknown")
    if test_set[col].isnull().any():
        test_set[col] = test_set[col].fillna("Unknown")

# Ensure categorical columns are strings
train_set[categorical_columns] = train_set[categorical_columns].astype(str)
test_set[categorical_columns] = test_set[categorical_columns].astype(str)

# Step 4: Encoding Categorical Variables
st.header("Step 4: Encoding Categorical Variables")

# Debugging: Display column names
st.write("### Columns in Train and Test Sets:")
st.write("Train Set Columns:", train_set.columns.tolist())
st.write("Test Set Columns:", test_set.columns.tolist())

# Ensure categorical columns are in the dataset
categorical_columns = ['Carrosserie', 'fuel_type', 'Gearbox']

missing_columns = [col for col in categorical_columns if col not in train_set.columns or col not in test_set.columns]
if missing_columns:
    st.error(f"The following columns are missing from the dataset: {missing_columns}")
    st.stop()

# Fill missing values in categorical columns
for col in categorical_columns:
    if train_set[col].isnull().any() or test_set[col].isnull().any():
        st.warning(f"Missing values detected in {col}. Filling with 'Unknown'.")
        train_set[col] = train_set[col].fillna('Unknown')
        test_set[col] = test_set[col].fillna('Unknown')

# Encode categorical columns
try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_train = encoder.fit_transform(train_set[categorical_columns])
    encoded_test = encoder.transform(test_set[categorical_columns])

    # Convert to DataFrame with proper column names
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_columns))
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_columns))

    # Concatenate encoded columns with the original datasets
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

#Step 6: Check
st.header("Step 6: Check")

# Ensure all columns in features are numeric
st.write("### Ensuring Numeric Columns in Features:")
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# Check for NaN values after ensuring numeric data
st.write("### Checking for NaN Values in Features Before Filling:")
st.write(f"NaN in X_train: {X_train.isnull().sum().sum()}")
st.write(f"NaN in X_test: {X_test.isnull().sum().sum()}")

# Fill NaN values with column mean
if X_train.isnull().any().any() or X_test.isnull().any().any():
    st.warning("NaN values detected in features. Filling with column mean.")
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())

# Debug Step: Display the updated feature sets
st.write("### Updated X_train:")
st.dataframe(X_train.head())

st.write("### Updated X_test:")
st.dataframe(X_test.head())

# Proceed with model training
st.write("### Proceeding with Model Training...")


# Step 7: Model Training
st.header("Step 7: Model Training")
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


# Step 8: Model Training
st.header("Step 8: Model Training")

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


# Step 9: Visualization
st.header("Step 9: Confusion Matrix")
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
