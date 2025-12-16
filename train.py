import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from data_acquisition import download_and_load_data, clean_data

def load_data():
    """Load and prepare data"""
    if os.path.exists("data/cleaned_heart_disease.csv"):
        df = pd.read_csv("data/cleaned_heart_disease.csv")
    else:
        df = download_and_load_data()
        df = clean_data(df)
    
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

def create_preprocessor(X):
    """Create preprocessing pipeline"""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    
    return ColumnTransformer(transformers, remainder='passthrough')

def train_model(model_name, model, X_train, X_test, y_train, y_test, preprocessor):
    """Train and evaluate a model with MLflow tracking"""
    
    with mlflow.start_run(run_name=model_name):
        # Create pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Log parameters
        mlflow.log_params(model.get_params())
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("cv_roc_auc_mean", cv_mean)
        mlflow.log_metric("cv_roc_auc_std", cv_std)
        
        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=f"HeartDiseaseClassifier_{model_name}"
        )
        
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"CV ROC-AUC: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        return pipeline, roc_auc

def hyperparameter_tuning(X_train, y_train, preprocessor):
    """Perform hyperparameter tuning for Random Forest"""
    
    with mlflow.start_run(run_name="RandomForest_Tuned"):
        # Define parameter grid
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5]
        }
        
        # Create pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42))
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Log best parameters and score
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_

def main():
    """Main training pipeline"""
    # Set MLflow experiment
    mlflow.set_experiment("heart_disease_prediction")
    
    # Load data
    X, y = load_data()
    print(f"Dataset shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create preprocessor
    preprocessor = create_preprocessor(X)
    
    # Train models
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        pipeline, score = train_model(name, model, X_train, X_test, y_train, y_test, preprocessor)
        if score > best_score:
            best_score = score
            best_model = pipeline
    
    # Hyperparameter tuning for best performing model
    tuned_model = hyperparameter_tuning(X_train, y_train, preprocessor)
    
    # Save best model
    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    
    print("\nTraining completed. Best model saved to models/best_model.pkl")

if __name__ == "__main__":
    main()