import numpy as np
import uvicorn
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load the pre-trained model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Input data model
class Flower(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: Flower):
    input_array = np.array([[data.sepal_length, data.sepal_width,
                            data.petal_length, data.petal_width]])
    pred = model.predict(input_array)
    return {"prediction": pred[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
