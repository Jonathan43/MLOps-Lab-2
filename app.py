import FastAPI
from simple_model import import_data, train_test, train_model, test_model


app = FastAPI()

@app.post("/predict")



uvicorn app:app --reload