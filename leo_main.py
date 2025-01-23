from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from leo.data_import import load_iris
from leo.data_split import splitting

# Load the Iris dataset
dataset = load_iris()

# Prepare features and target
X = dataset.iloc[:, :-1]  # Features
y = dataset.iloc[:, -1]   # Target class

# Split the data
X_train, X_test, y_train, y_test = splitting(X, y)

# Create classification pipeline
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))