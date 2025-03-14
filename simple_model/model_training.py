import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from simple_model.custom_train_test import train_test
from simple_model.importing_data import import_data


def train_model(dataset):
    X_train, X_test, y_train, y_test = train_test(dataset)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model
