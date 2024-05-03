import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from scipy.io import arff
import pandas as pd


#This is where you decide what dataset you want to use
#If you want to use an openml dataset, uncomment the 3 lines below and alter the dataset name in the first field to what you want to use
#Used datasets: "image", "yeast", "emotions", "reueters"

#X, Y = fetch_openml("yeast", version=4, return_X_y=True)
#Y = Y == "TRUE"
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Keep the next 4 lines uncommented if you want to use the PC dataset
X_train =  pd.read_csv('cc_x_train.csv')
Y_train = pd.read_csv('cc_y_train.csv')
X_test =  pd.read_csv('cc_x_test.csv')
Y_test = pd.read_csv('cc_y_test.csv')

#This is where we create our base algorithms for the classifier chains
base_lr = LogisticRegression(max_iter=15000)
mydecision = DecisionTreeClassifier()

#OVR testing
ovr = OneVsRestClassifier(base_lr)
ovr.fit(X_train, Y_train)
Y_pred_ovr = ovr.predict(X_test)
ovr_jaccard_score = jaccard_score(Y_test, Y_pred_ovr >= 0.5, average="samples")

#Change the first field if you want to use a different algorithm
#Lines 42-51 create the chains, fit them to the training data, and then predict the test data. We then get the jaccard for each chain
chains = [ClassifierChain(base_lr, order="random", random_state=i+61) for i in range(10)]
for chain in chains:
    chain.fit(X_train, Y_train)


Y_pred_chains = np.array([chain.predict(X_test) for chain in chains])
chain_jaccard_scores = [
    jaccard_score(Y_test, Y_pred_chain >= 0.5, average="samples")
    for Y_pred_chain in Y_pred_chains
]

#This is where we create the ensemble chain and predict the test data. We then get the jaccard and accuracy scores for the ensemble chain
#Change the weighting for prediction resolution by changing the number after the >=, must be between 0 and 1

Y_pred_ensemble = Y_pred_chains.mean(axis=0)
ensemble_jaccard_score = jaccard_score(
    Y_test, Y_pred_ensemble >= 0.8, average="samples"
)

#Where we analyze the results vs OVR
print("DATASET IMPROVEMENT: ")
print((ensemble_jaccard_score-ovr_jaccard_score))
print((ensemble_jaccard_score-ovr_jaccard_score)/ovr_jaccard_score)
print("OVR Jaccard Score: ", ovr_jaccard_score)
model_scores = [ovr_jaccard_score] + chain_jaccard_scores + [ensemble_jaccard_score]

#GRAPHING THE RESULTS
model_names = (
    "OVR",
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
ax.set_title("Classifier Chain Ensemble Performance Comparison - Logistic Regression - PC Dataset")
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation="vertical")
ax.set_ylabel("Jaccard Similarity Score")
ax.set_ylim([min(model_scores) * 0.9, max(model_scores) * 1.1])
colors = ["r"] + ["b"] * len(chain_jaccard_scores) + ["g"]
ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
plt.tight_layout()
plt.show()