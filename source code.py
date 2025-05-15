import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




try:
    df = pd.read_csv('mock_creditcard.csv')
    display(df.head())
    print(df.shape)
except FileNotFoundError:
    print("Error: 'mock_creditcard.csv' not found. Please ensure the file exists in the current directory.")
    # 1. Examine data types
print("Data Types:\n", df.dtypes)

# 2. Check for missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
print("\nMissing Values:\n", missing_values)
print("\nMissing Value Percentage:\n", missing_percentage)

# 3. Descriptive statistics for numerical features
numerical_features = df.select_dtypes(include=['number'])
print("\nDescriptive Statistics:\n", numerical_features.describe())

# 4. Analyze target variable distribution
print("\nTarget Variable Distribution:\n", df['Class'].value_counts())
plt.figure(figsize=(8, 6))
df['Class'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Fraudulent vs. Non-Fraudulent Transactions')
plt.xlabel('Class (0: Non-Fraudulent, 1: Fraudulent)')
plt.ylabel('Number of Transactions')
plt.show()

# 5. Explore feature relationships with the target variable
print("\nCorrelation Matrix:\n", numerical_features.corrwith(df['Class']))
plt.figure(figsize=(12, 10))
sns.heatmap(numerical_features.corr(), annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()
# 1. Examine data types
print("Data Types:\n", df.dtypes)

# 2. Check for missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
print("\nMissing Values:\n", missing_values)
print("\nMissing Value Percentage:\n", missing_percentage)

# 3. Descriptive statistics for numerical features
numerical_features = df.select_dtypes(include=['number'])
print("\nDescriptive Statistics:\n", numerical_features.describe())

# 4. Analyze target variable distribution
print("\nTarget Variable Distribution:\n", df['Class'].value_counts())
plt.figure(figsize=(8, 6))
df['Class'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Fraudulent vs. Non-Fraudulent Transactions')
plt.xlabel('Class (0: Non-Fraudulent, 1: Fraudulent)')
plt.ylabel('Number of Transactions')
plt.show()

# 5. Explore feature relationships with the target variable
print("\nCorrelation Matrix:\n", numerical_features.corrwith(df['Class']))
plt.figure(figsize=(12, 10))
sns.heatmap(numerical_features.corr(), annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()
# 6. Determine the shape of the DataFrame
print("\nDataFrame Shape:", df.shape)

# Handle Outliers using IQR
numerical_features = ['V' + str(i) for i in range(1, 29)] + ['Amount']
for col in numerical_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# Check for Missing Values again
missing_values = df.isnull().sum()
print("Missing Values after outlier handling:\n", missing_values)

# Impute Missing Values (if any)
for col in numerical_features:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

print(df.isnull().sum().sum())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# 2. Check for and handle categorical features (none found in this dataset)
# All columns appear to be numerical based on the initial data exploration.

# 3. Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix
# Predict on the test set
y_pred = model.predict(X_test)

# Generate and print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Generate and print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
from sklearn.metrics import confusion_matrix
# Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Visualize feature importances
feature_importances = pd.DataFrame({'Feature': df.drop('Class', axis=1).columns,
                                    'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.show()

# Optional visualization: Distribution of key features
# Example: Amount distribution for fraudulent and non-fraudulent transactions
plt.figure(figsize=(8, 6))
sns.histplot(df[df['Class'] == 0]['Amount'], color='skyblue', label='Non-Fraud', kde=True)
sns.histplot(df[df['Class'] == 1]['Amount'], color='salmon', label='Fraud', kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.legend()
plt.show()
