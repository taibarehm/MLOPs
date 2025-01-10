import os
import joblib
import numpy as np

# Path to the saved model
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
    sample_data = np.array([
        [0.03807591, 0.05068012, 0.06169621, 0.02187235, -0.0442235,
         -0.03482076, -0.04340085, -0.00259226, 0.01990842, -0.01764613],
        [0.02717829, -0.04464164, -0.01350402, -0.01599899, -0.00286131,
         -0.01944209, -0.06899065, -0.00259226, 0.00286131, -0.02593034]
    ])

    # Make predictions
    predictions = model.predict(sample_data)
    assert len(predictions) == len(sample_data), "Prediction length mismatch!"
    