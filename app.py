import numpy as np
import pandas as pd
from fastapi import FastAPI
from simple_model.importing_data import import_data
from simple_model.model_training import train_model
from simple_model.model_inference import prediction

# Training the model

dataset = import_data()
model = train_model(dataset)

app = FastAPI()

@app.post("/predict", response_model=prediction, model = model)
def predict(data: datapoint):
    # Use the loaded model to make a prediction
    prediction = prediction(model, datapoint)
    return {"prediction": int(prediction[0])}

# Run the FastAPI app on localhost
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)