import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def download_and_load_data():
    """Download and load heart disease dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"]
    
    df = pd.read_csv(url, names=cols)
    
    # Save raw data
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_heart_disease.csv", index=False)
    
    return df

def clean_data(df):
    """Clean and preprocess the data"""
    # Replace '?' with NaN
    df.replace("?", pd.NA, inplace=True)
    
    # Convert numeric columns
    numeric_cols = ["age","trestbps","chol","thalach","oldpeak","ca","thal"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Convert target to binary
    df["target"] = (df["target"] > 0).astype(int)
    
    # Save cleaned data
    df.to_csv("data/cleaned_heart_disease.csv", index=False)
    
    return df

def perform_eda(df):
    """Perform comprehensive EDA"""
    os.makedirs("plots", exist_ok=True)
    
    # Basic statistics
    print("Dataset Shape:", df.shape)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Class distribution
    plt.figure(figsize=(8, 6))
    df['target'].value_counts().plot(kind='bar')
    plt.title('Heart Disease Distribution')
    plt.xlabel('Heart Disease (0=No, 1=Yes)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('plots/class_distribution.png')
    plt.close()
    
    # Age distribution by target
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age', hue='target', bins=20, alpha=0.7)
    plt.title('Age Distribution by Heart Disease')
    plt.tight_layout()
    plt.savefig('plots/age_distribution.png')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()
    
    # Feature distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    
    for i, feature in enumerate(numeric_features):
        row, col = i // 3, i % 3
        sns.boxplot(data=df, x='target', y=feature, ax=axes[row, col])
        axes[row, col].set_title(f'{feature} by Heart Disease')
    
    # Remove empty subplots
    for i in range(len(numeric_features), 9):
        row, col = i // 3, i % 3
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.savefig('plots/feature_distributions.png')
    plt.close()
    
    print("\nEDA plots saved to 'plots/' directory")

if __name__ == "__main__":
    # Download and load data
    df = download_and_load_data()
    print("Data downloaded successfully")
    
    # Clean data
    df_clean = clean_data(df)
    print("Data cleaned successfully")
    
    # Perform EDA
    perform_eda(df_clean)
    print("EDA completed successfully")