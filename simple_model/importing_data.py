import numpy as np
import pandas as pd

def import_data():
    # Import the data from the file
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    return dataset