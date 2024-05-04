import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score

# Load datasets
X_train = pd.read_csv('PC dataset/cc_x_train-2.csv')
X_test = pd.read_csv('PC dataset/cc_x_test.csv')
Y_train = pd.read_csv('PC dataset/cc_y_train-2.csv')
Y_test = pd.read_csv('PC dataset/cc_y_test.csv')

# Optional: Analyze correlations to decide on label pairs if necessary
#correlations = pd.read_csv('cc_correlations.csv')


def add_calibration_label(Y):
    calibration_label = np.zeros((Y.shape[0], 1), dtype=int)
    return np.hstack((Y, calibration_label))


def multilabel_to_calibrated_ranking(Y):
    n_labels = Y.shape[1]
    label_combinations = [(i, n_labels - 1) for i in range(n_labels - 1)]
    Y_pairs = np.zeros((Y.shape[0], len(label_combinations)))
    for idx, (label1, label2) in enumerate(label_combinations):
        Y_pairs[:, idx] = (Y[:, label1] > Y[:, label2]).astype(int)
    return Y_pairs


Y_train_with_cal = add_calibration_label(Y_train)
Y_test_with_cal = add_calibration_label(Y_test)
Y_train_pairs = multilabel_to_calibrated_ranking(Y_train_with_cal)
Y_test_pairs = multilabel_to_calibrated_ranking(Y_test_with_cal)

models = {
    'Logistic Regression': LogisticRegression(max_iter=15000)
    #'Decision Tree': DecisionTreeClassifier(), # Uncomment if you want to use the Decision Tree model
}

model_results = {}
for model_name, model in models.items():
    jaccard_scores = []
    for i in range(Y_train_pairs.shape[1]):
        # Check if the label pair has at least two classes
        if len(np.unique(Y_train_pairs[:, i])) > 1:
            model.fit(X_train, Y_train_pairs[:, i])
            y_pred = model.predict(X_test)
            jaccard = jaccard_score(
                Y_test_pairs[:, i], y_pred, average='binary', zero_division=1)
            jaccard_scores.append(jaccard)
    model_results[model_name] = {'Jaccard Scores': jaccard_scores}

for model_name, scores in model_results.items():
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(scores['Jaccard Scores']))
    plt.bar(indices, scores['Jaccard Scores'], color='green',
            alpha=0.6, label='Jaccard per Label Pair')
    total_jaccard = np.mean(
        scores['Jaccard Scores']) if scores['Jaccard Scores'] else 0
    plt.bar(len(indices), total_jaccard, color='purple',
            label='Total Ensemble Jaccard')
    plt.xlabel('Label Pair Index')
    plt.ylabel('Jaccard Score')
    plt.title(
        f'Jaccard Scores for Each Label Pair and Total Ensemble - {model_name}')
    plt.xticks(ticks=list(indices) + [len(indices)],
               labels=list(indices) + ['Ensemble'])
    plt.legend()
    plt.show()

print("Jaccard score:", total_jaccard)
