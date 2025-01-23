import FastAPI
from simple_model.importing_data import import_data
from simple_model.custom_train_test import train_test
from simple_model.model import train_model, test_model


app = FastAPI()

@app.post("/predict")



uvicorn app:app --reload