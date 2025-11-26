# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from google.colab import files
import zipfile
import io

# File Upload
uploaded = files.upload()
for file in uploaded:
    if file.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(uploaded[file]), 'r') as zip_ref:
            zip_ref.extractall('/content/')
        print("Zip extracted.")

# Load Dataset (Replace with actual CSV name after checking extracted files)
df = pd.read_csv('/content/WA_Fn-UseC_-Telco-Customer-Churn.csv')  # Change if needed

# Drop customerID
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

# Basic Info
print("Head:\n", df.head(), "\n")
print("Tail:\n", df.tail(), "\n")

# Clean total charges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    if col != 'Churn':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Descriptive Statistics
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
stats_summary = pd.DataFrame(columns=['Mean', 'Median', 'Mode', 'Std Dev', 'Variance'])

for col in numeric_cols:
    stats_summary.loc[col] = [
        df[col].mean(),
        df[col].median(),
        df[col].mode()[0],
        df[col].std(),
        df[col].var()
    ]
print("Descriptive Statistics:\n", stats_summary, "\n")

# Skewness and Kurtosis
for col in numeric_cols:
    s = skew(df[col])
    k = kurtosis(df[col])
    skew_type = 'Positive Skew' if s > 0 else 'Negative Skew' if s < 0 else 'Symmetrical'
    kurt_type = 'Leptokurtic' if k > 0 else 'Platykurtic' if k < 0 else 'Mesokurtic'
    print(f"{col} — Skew: {s:.2f} ({skew_type}), Kurtosis: {k:.2f} ({kurt_type})")
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'{col} — {skew_type}, {kurt_type}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Features and Target
X = df.drop(columns='Churn')
y = df['Churn']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Evaluate
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# User Input Prediction
print("\n--- New User Prediction ---")
user_data = {}
for col in X.columns:
    if col in label_encoders:
        options = label_encoders[col].classes_
        value = input(f"{col} {list(options)}: ")
        user_data[col] = label_encoders[col].transform([value])[0]
    else:
        value = float(input(f"{col}: "))
        user_data[col] = value

user_df = pd.DataFrame([user_data])
user_scaled = scaler.transform(user_df)
user_prob = knn.predict_proba(user_scaled)
user_pred = knn.predict(user_scaled)

print("\nPrediction:", "Yes" if user_pred[0] == 1 else "No")
print("Probability: {:.2f}%".format(np.max(user_prob) * 100))