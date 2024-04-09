import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.datasets import make_multilabel_classification
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


###X = np.array(ct.fit_transform(X), dtype = str)


#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

Y_train = pd.read_csv('CC_LABELS2.csv')



X_train =  pd.read_csv('CC.csv')
Y_test = pd.read_csv('CC_LABELS_test.csv')



X_test =  pd.read_csv('CC_test.csv')
#ohe=OneHotEncoder()
#ct=ColumnTransformer(transformers=[('encoder',ohe,[0])],remainder='passthrough')
#print(X)
#print(Y)
#X_train=np.array(ct.fit_transform(X))
#Y_train=np.array(ct.fit_transform(Y))
print(X_train)
X_test = pd.read_csv('CC_TEST.csv')
base_lr = LogisticRegression(max_iter=15000)
chains = [ClassifierChain(base_lr, order="random", random_state=i) for i in range(10)]
for chain in chains:
    chain.fit(X_train, Y_train)

Y_pred_chains = np.array([chain.predict_proba(X_test) for chain in chains])

chain_jaccard_scores = [
    accuracy_score(Y_test, Y_pred_chain >= 0.5)
    for Y_pred_chain in Y_pred_chains
]

Y_pred_ensemble = Y_pred_chains.mean(axis=0)
ensemble_jaccard_score = accuracy_score(Y_test, Y_pred_ensemble >= 0.5)
model_scores = chain_jaccard_scores + [ensemble_jaccard_score]

model_names = (
    "Chain 1",
    "Chain 2",
    "Chain 3",
    "Chain 4",
    "Chain 5",
    "Chain 6",
    "Chain 7",
    "Chain 8",
    "Chain 9",
    "Chain 10",
    "Ensemble",
)

x_pos = np.arange(len(model_names))

fig, ax = plt.subplots(figsize=(7, 4))
ax.grid(True)
ax.set_title("Classifier Chain Ensemble Performance Comparison")
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation="vertical")
ax.set_ylabel("Jaccard Similarity Score")
ax.set_ylim([min(model_scores) * 0.9, max(model_scores) * 1.1])
colors = ["r"] + ["b"] * len(chain_jaccard_scores) + ["g"]
ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
plt.tight_layout()
plt.show()