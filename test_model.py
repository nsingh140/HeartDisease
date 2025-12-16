import pytest
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.pipeline import Pipeline
from data_acquisition import download_and_load_data, clean_data
from train import create_preprocessor, load_data

class TestDataProcessing:
    
    def test_data_download(self):
        """Test data download functionality"""
        df = download_and_load_data()
        assert not df.empty
        assert df.shape[1] == 14  # 13 features + 1 target
    
    def test_data_cleaning(self):
        """Test data cleaning functionality"""
        # Create sample data with missing values (matching expected columns)
        data = {
            'age': [63, 37, '?'],
            'sex': [1, 1, 0],
            'cp': [3, 2, 1],
            'trestbps': [145, 130, '?'],
            'chol': [233, 250, 200],
            'fbs': [1, 0, 1],
            'restecg': [0, 1, 0],
            'thalach': [150, 187, 172],
            'exang': [0, 0, 1],
            'oldpeak': [2.3, 3.5, '?'],
            'slope': [0, 0, 2],
            'ca': [0, 0, '?'],
            'thal': [1, 2, 3],
            'target': [1, 0, 1]
        }
        df = pd.DataFrame(data)
        
        # Clean data
        df_clean = clean_data(df.copy())
        
        # Check no missing values
        assert df_clean.isnull().sum().sum() == 0
        # Check target is binary
        assert set(df_clean['target'].unique()).issubset({0, 1})
        # Check data shape (should have fewer rows after dropping NaN)
        assert df_clean.shape[0] <= df.shape[0]
    
    def test_preprocessor_creation(self):
        """Test preprocessor creation"""
        X, y = load_data()
        preprocessor = create_preprocessor(X)
        
        # Test fit and transform
        X_transformed = preprocessor.fit_transform(X)
        assert X_transformed.shape[0] == X.shape[0]

class TestModel:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return {
            'age': 63,
            'sex': 1,
            'cp': 3,
            'trestbps': 145,
            'chol': 233,
            'fbs': 1,
            'restecg': 0,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 2.3,
            'slope': 0,
            'ca': 0,
            'thal': 1
        }
    
    def test_model_loading(self):
        """Test model loading"""
        if os.path.exists("models/best_model.pkl"):
            with open("models/best_model.pkl", "rb") as f:
                model = pickle.load(f)
            assert model is not None
            assert hasattr(model, 'predict')
    
    def test_model_prediction(self, sample_data):
        """Test model prediction"""
        if os.path.exists("models/best_model.pkl"):
            with open("models/best_model.pkl", "rb") as f:
                model = pickle.load(f)
            
            # Create DataFrame
            df = pd.DataFrame([sample_data])
            
            # Make prediction
            prediction = model.predict(df)
            
            # Check prediction format
            assert len(prediction) == 1
            assert prediction[0] in [0, 1]
    
    def test_model_prediction_probability(self, sample_data):
        """Test model prediction probability"""
        if os.path.exists("models/best_model.pkl"):
            with open("models/best_model.pkl", "rb") as f:
                model = pickle.load(f)
            
            df = pd.DataFrame([sample_data])
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(df)
                assert proba.shape == (1, 2)
                assert np.allclose(proba.sum(axis=1), 1.0)

class TestAPI:
    
    def test_input_validation(self):
        """Test input validation"""
        from app import HeartDiseaseInput
        
        # Valid input
        valid_data = {
            'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145,
            'chol': 233, 'fbs': 1, 'restecg': 0, 'thalach': 150,
            'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        }
        
        input_obj = HeartDiseaseInput(**valid_data)
        assert input_obj.age == 63
        assert input_obj.sex == 1
    
    def test_invalid_input(self):
        """Test invalid input handling"""
        from app import HeartDiseaseInput
        from pydantic import ValidationError
        
        # Missing required field
        invalid_data = {'age': 63, 'sex': 1}
        
        with pytest.raises(ValidationError):
            HeartDiseaseInput(**invalid_data)

if __name__ == "__main__":
    pytest.main([__file__])