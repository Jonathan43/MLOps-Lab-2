# Simple classification model for the Iris dataset
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from custom_train_test import train_test
from custom_train_test import train_test
from importing_data import import_data

def train_model(dataset):
    X_train, X_test, y_train, y_test = train_test(dataset)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model
