import FastAPI
from simple_model.importing_data import import_data
from simple_model.custom_train_test import train_test
from simple_model.model import train_model, test_model


app = FastAPI()

@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    # Use the loaded model to make a prediction
    prediction = model.predict([[data.feature1, data.feature2]])
    return {"prediction": int(prediction[0])}

# Run the FastAPI app on localhost
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)