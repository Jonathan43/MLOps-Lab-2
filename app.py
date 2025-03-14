import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from simple_model.importing_data import import_data
from simple_model.model_training import train_model
from simple_model.model_inference import prediction

# Training the model
dataset = import_data()
model = train_model(dataset)

app = FastAPI()

# We define the input data format
class Flower(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict(data: Flower):
    input_array = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    pred = prediction(model, input_array)
    return {"prediction": pred[0]}

# Run the FastAPI app on localhost
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)