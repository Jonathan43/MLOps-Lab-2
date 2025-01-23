import subprocess
import sys

try:
    import fastapi
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi"])

from fastapi import FastAPI
from simple_model.importing_data import import_data
from simple_model.custom_train_test import train_test
from simple_model.model import train_model, test_model

from simple_model.model import prediction

app = FastAPI()

@app.post("/predict", response_model=prediction)
def predict(data: import_data()):
    # Use the loaded model to make a prediction
    prediction = prediction(model, dataset)
    return {"prediction": int(prediction[0])}

# Run the FastAPI app on localhost
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)