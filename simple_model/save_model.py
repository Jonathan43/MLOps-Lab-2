import pickle
from simple_model.importing_data import import_data
from simple_model.model_training import train_model

# Train and save the model
dataset = import_data()
model = train_model(dataset)

# Save to pickle file
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)