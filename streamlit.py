import streamlit as st
import pandas as pd
import numpy as np

st.title("Credit Card Fraud Detection Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    st.write("### Data Types")
    st.write(df.dtypes)

    st.write("### Missing Values")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    st.write(missing_values)
    st.write("Percentage Missing:")
    st.write(missing_percentage)

    st.write("### Descriptive Statistics")
    numerical_features = df.select_dtypes(include=['number'])
    st.write(numerical_features.describe())

    st.write("### Target Variable Distribution")
    st.bar_chart(df['Class'].value_counts())

    st.write("### Correlation Matrix")
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(numerical_features.corr(), cmap='coolwarm', ax=ax1)
    st.pyplot(fig1)

    # Handle Outliers with IQR
    st.write("Handling outliers using IQR...")
    feature_cols = ['V' + str(i) for i in range(1, 29)] + ['Amount']
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # Impute missing values
    for col in feature_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    st.write("### Total Missing Values After Handling:")
    st.write(df.isnull().sum().sum())

    