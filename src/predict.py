import joblib
from sklearn.datasets import load_iris

# Load trained model
model = joblib.load("models/iris_model.pkl")

# Load dataset to get class names
iris = load_iris()
class_names = iris.target_names

# Example flower measurements
sample_flower = [[5.1, 3.5, 1.4, 0.2]]

# Predict
prediction = model.predict(sample_flower)

# Convert number to flower name
flower_name = class_names[prediction[0]]

print("Predicted flower:", flower_name)