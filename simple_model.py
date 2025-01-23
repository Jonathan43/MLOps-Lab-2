# Simple classification model for the Iris dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from importing_data import import_data
from custom_train_test import train_test

# Download and import data
dataset = import_data()

# Train test split
X_train, X_test, y_train, y_test = train_test(dataset)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy