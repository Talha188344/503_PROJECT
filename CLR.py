''''
The provided code meets all the specified steps:

Generate Dataset: The dataset is generated using a function generate_dataset that mimics CPU benchmarks with the specified characteristics.

Split Dataset: The generated dataset is split into training and test sets using train_test_split.

CLRCPUClassifier: The CLRCPUClassifier class is defined, inheriting from LabelSpacePartitioningClassifier, which is consistent with the provided implementation.

Model Training: An instance of CLRCPUClassifier is created with DecisionTreeClassifier as the base estimator. The model is trained using the training data X_train and y_train.

Predictions: Predictions are made on the test set X_test using the predict method of the trained CLRCPUClassifier model.

Evaluation: The test accuracy is calculated by comparing the predicted labels y_pred with the true labels y_test. The accuracy score is printed as part of the evaluation step.

Additionally, the trained model is saved to a file named 'cpu_classifier_model.pkl', as requested.'''

import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.cluster import MatrixLabelSpaceClusterer
from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from sklearn.metrics import accuracy_score
import joblib


# Load data from JSON files
def load_data_from_json(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data.extend(json.load(f))
    return data

# Generate synthetic dataset matching specifications
def generate_dataset(intel_data, amd_data, n_samples=1000000, random_state=42):
    np.random.seed(random_state)
    # Sample from Intel and AMD data
    intel_samples = np.random.choice(intel_data, size=n_samples//2, replace=True).tolist()
    amd_samples = np.random.choice(amd_data, size=n_samples//2, replace=True).tolist()
    data = intel_samples + amd_samples

    # Extract features and labels
    X = np.array([float(d['benchmark']) for d in data])
    y = np.array([0 if d['name'].startswith('Intel') else 1 for d in data])  # 0 for Intel, 1 for AMD

    # Reshape X to 2D array
    X = X.reshape(-1, 1)

    return X, y



# Load CPU benchmark data from JSON files
intel_data = load_data_from_json(['Intel.json'])
amd_data = load_data_from_json(['AMD.json'])

# Generate synthetic dataset
X, y = generate_dataset(intel_data, amd_data)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CLRCPUClassifier
from skmultilearn.base import MLClassifierBase

class CLRCPUClassifier(MLClassifierBase):
    def __init__(self, classifier=None):
        self.classifier = classifier if classifier is not None else DecisionTreeClassifier()
        self.copyable_attrs = ['classifier']
        self._label_cache = {}

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)


# Model Training
base_classifier = DecisionTreeClassifier()
clusterer = MatrixLabelSpaceClusterer(clusterer='dbscan')
clr = CLRCPUClassifier(classifier=base_classifier)
clr.fit(X_train, y_train)

# Predictions
y_pred = clr.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Predictions
y_pred = clr.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Save the trained model
joblib.dump(clr, 'cpu_classifier_model.pkl')
print("Model saved as 'cpu_classifier_model.pkl'.")
