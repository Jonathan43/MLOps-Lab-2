# Simple classification model for the Iris dataset
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from model_training import train_model

def prediction(model, datapoint):
    return model.predict(datapoint)