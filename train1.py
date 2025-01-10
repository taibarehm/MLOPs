# train.py

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Load the diabetes dataset
print("Loading the diabetes dataset.")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Split the dataset into training and testing sets
print("Splitting the data into training and testing sets.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
print("Training the Linear Regression model.")
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
print("Evaluating the model.")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Save the trained model to the 'models' directory
print("Saving the trained model.")
os.makedirs("models", exist_ok=True)
model_path = "C:/Users/asus/OneDrive/Desktop/MLOPs/linear_regression_diabetes.pkl"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
print("Model trained and saved!")
