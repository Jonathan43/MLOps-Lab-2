import FastAPI
import simple_model


app = FastAPI()

@app.post("/predict")



uvicorn app:app --reload