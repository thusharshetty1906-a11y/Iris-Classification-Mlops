from fastapi import FastAPI
import joblib

app = FastAPI()

# Load trained model
model = joblib.load("models/iris_model.pkl")


@app.get("/")
def home():
    return {"message": "Iris Prediction API is running"}


@app.get("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):

    data = [[sepal_length, sepal_width, petal_length, petal_width]]

    prediction = model.predict(data)

    return {"prediction": int(prediction[0])}