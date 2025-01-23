import pandas as pd

def load_iris():
    # Load the Iris dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    return pd.read_csv(url, names=names)
