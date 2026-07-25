import pandas as pd
import numpy as np
import joblib
import os

def load_model(model_path="models/churn_model.pkl"):
    """Load the trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    return joblib.load(model_path)

def predict_churn(model, input_data):
    """Predict churn for a single customer."""
    # Get probability
    probability = model.predict_proba(input_data)[0][1]
    
    # Get prediction
    prediction = model.predict(input_data)[0]
    
    return prediction, probability

def batch_predict(model, input_data):
    """Predict churn for multiple customers."""
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]
    
    results = pd.DataFrame({
        "prediction": predictions,
        "churn_probability": probabilities
    })
    
    results["churn_risk"] = results["churn_probability"].apply(
        lambda x: "High" if x >= 0.5 else "Low"
    )
    
    return results

if __name__ == "__main__":
    # Example usage
    model = load_model()
    print("Model loaded successfully!")

