from fastapi import FastAPI
from simple_model.importing_data import import_data
from simple_model.custom_train_test import train_test
from simple_model.model_inference import train_model, test_model

from simple_model.model_inference import prediction

app = FastAPI()




@app.post("/predict", response_model=prediction)
def predict(data: datapoint):
    # Use the loaded model to make a prediction
    prediction = prediction(model, datapoint)
    return {"prediction": int(prediction[0])}

# Run the FastAPI app on localhost
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)