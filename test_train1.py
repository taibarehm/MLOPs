import os
import joblib
import numpy as np

# Path to the saved model
#  MODEL_PATH = "E:/MLOps_Assignment1/models/diabetes_model.pkl"
MODEL_PATH = "models/linear_regression_diabetes.pkl"

def test_model_file_exists():
    """
    Test if the model file has been saved.
    """
    assert os.path.exists(MODEL_PATH), "Model file not found!"

def test_model_predictions():
    """
    Test if the model can make predictions on sample data.
    """
    # Load the model
    model = joblib.load(MODEL_PATH)
    
    # Create sample data (2 samples from the diabetes dataset)
    sample_data = np.array([[0.03807591,  0.05068012,  0.06169621,  0.02187235, -0.0442235, 
                             -0.03482076, -0.04340085, -0.00259226,  0.01990842, -0.01764613],
                            [-0.00188202, -0.04464164, -0.05147406, -0.02632702, -0.00844872, 
                             -0.01916334,  0.07441156, -0.03949338, -0.06832974, -0.09220405]])
    
    # Predict using the model
    predictions = model.predict(sample_data)
    
    # Check if predictions are numeric and have the correct shape
    assert len(predictions) == 2, "Model did not return predictions for 2 samples!"
    print("Model predictions test passed!")

# Run the tests
if __name__ == "__main__":
    test_model_file_exists()
    test_model_predictions()
    print("All tests passed!")
