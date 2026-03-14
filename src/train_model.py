import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start MLflow experiment
mlflow.start_run()

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Log metrics
mlflow.log_metric("accuracy", accuracy)

# Log model
mlflow.sklearn.log_model(model, "iris_model")

# Save model
joblib.dump(model, "models/iris_model.pkl")

mlflow.end_run()

print("Model saved successfully!")