import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def train_test(dataset):
    X = dataset.iloc[:, 0:4].values
    y = dataset.iloc[:, 4].values
    return train_test_split(X, y, test_size=0.2, random_state=0)