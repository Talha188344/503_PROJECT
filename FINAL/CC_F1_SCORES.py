import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.datasets import make_multilabel_classification
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import pandas as pd


#This is where you decide what dataset you want to use
#If you want to use an openml dataset, uncomment the 3 lines below and alter the dataset name in the first field to what you want to use
#Used datasets: "image", "yeast", "emotions", "reueters"

X, Y = fetch_openml("reuters", version=4, return_X_y=True)
Y = Y == "TRUE"
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Uncomment the next 4 lines if you want to use the PC dataset
#X_train =  pd.read_csv('cc_x_train.csv')
#Y_train = pd.read_csv('cc_y_train.csv')
#X_test =  pd.read_csv('cc_x_test.csv')
#Y_test = pd.read_csv('cc_y_test.csv')

#This is where we create our base algorithms for the classifier chains
base_lr = LogisticRegression(max_iter=15000)
mysvc = SVC()
ovr = OneVsRestClassifier(base_lr)
mydecision = DecisionTreeClassifier()
#OVR testing
ovr.fit(X_train, Y_train)
Y_pred_ovr = ovr.predict(X_test)
ovr_jaccard_score = jaccard_score(Y_test, Y_pred_ovr, average="samples")
ovr_accuracy_score = accuracy_score(Y_test, Y_pred_ovr)
ovr_f1_score = f1_score(Y_test, Y_pred_ovr, average=None)


#Change the first field if you want to use a different algorithm
#Lines 51-64 create the chains, fit them to the training data, and then predict the test data. We then get the jaccard and accuracy scores for each chain
chains = [ClassifierChain(base_lr, order="random", random_state=i+92) for i in range(12)]
for chain in chains:
    chain.fit(X_train, Y_train)

Y_pred_chains = np.array([chain.predict(X_test) for chain in chains])

chain_jaccard_scores = [
    jaccard_score(Y_test, Y_pred_chain >= 0.5, average="samples")
    for Y_pred_chain in Y_pred_chains
]
chain_accuracy_scores = [
    accuracy_score(Y_test, Y_pred_chain >= 0.5)
    for Y_pred_chain in Y_pred_chains
]

#This is where we create the ensemble chain and predict the test data. We then get the jaccard and accuracy scores for the ensemble chain
Y_pred_ensemble = Y_pred_chains.mean(axis=0)
#Change the weighting for prediction resolution by changing the number after the >=, must be between 0 and 1
ensemble_accuracy_score = accuracy_score(Y_test, Y_pred_ensemble >= 0.5)
ensemble_jaccard_score = jaccard_score(Y_test, Y_pred_ensemble >= 0.5, average="samples")
f1_score_temp = f1_score(Y_test, Y_pred_ensemble >= 0.5, average=None)

f1_score_temp = f1_score(Y_test, Y_pred_ensemble >= 0.5, average=None)
#model_scores = [ovr_jaccard_score] + [ovr_accuracy_score] + [ovr_f1_score[0]] + [ovr_f1_score[1]] + [ovr_f1_score[2]] + [chain_jaccard_scores[0]] + [chain_accuracy_scores[0]] + [chain_jaccard_scores[1]] + [chain_accuracy_scores[1]] + [chain_jaccard_scores[2]] + [chain_accuracy_scores[2]] + [chain_jaccard_scores[3]] + [chain_accuracy_scores[3]] + [chain_jaccard_scores[4]] + [chain_accuracy_scores[4]] + [chain_jaccard_scores[5]] + [chain_accuracy_scores[5]] + [ensemble_jaccard_score] + [ensemble_accuracy_score] + [f1_score_temp[0]] + [f1_score_temp[1]] + [f1_score_temp[2]]
model_scores = [ovr_f1_score[qs] for qs in range(len(ovr_f1_score))] + [f1_score_temp[xs] for xs in range(len(f1_score_temp))]
model_scores = [ovr_jaccard_score] + chain_jaccard_scores + [ensemble_jaccard_score]

#Graphing the results based on what data you want to display, currently configured for F1 scores

#model_names = (
#    "OVR Jaccard",
#    "OVR Accuracy",
#    "OVR F1 Label 1",
#    "OVR F1 Label 2",
#    "OVR F1 Label 3",
#    "Chain 1 Jaccard",
#    "Chain 1 Accuracy",
#    "Chain 2 Jaccard",
#    "Chain 2 Accuracy",
#    "Chain 3 Jaccard",
#    "Chain 3 Accuracy",
#    "Chain 4 Jaccard",
#    "Chain 4 Accuracy",
#    "Chain 5 Jaccard",
#    "Chain 5 Accuracy",
#    "Chain 6 Jaccard",
#    "Chain 6 Accuracy",
#    "Ensemble Jaccard",
#    "Ensemble Accuracy",
#    "Ensemble F1 Label 1",
#    "Ensemble F1 Label 2",
#    "Ensemble F1 Label 3",
#)
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
    "Chain 11",
    "Chain 12",
    "Ensemble",
    
)
#model_names = (
#    "OVR Label 1 F1 Score",
#    "OVR Label 2 F1 Score",
#    "OVR Label 3 F1 Score",
#    "OVR Label 4 F1 Score",
#    "OVR Label 5 F1 Score",
#    "OVR Label 6 F1 Score",
#    "OVR Label 7 F1 Score",
#    "Ensemble Label 1 F1 Score",
#    "Ensemble Label 2 F1 Score",
#    "Ensemble Label 3 F1 Score",
#    "Ensemble Label 4 F1 Score",
#    "Ensemble Label 5 F1 Score",
#    "Ensemble Label 6 F1 Score",
#    "Ensemble Label 7 F1 Score",
#    
#    
#)
x_pos = np.arange(len(model_names))

fig, ax = plt.subplots(figsize=(7, 4))
ax.grid(True)
ax.set_title("Classifier Chain Order Performance Comparison - PC Benchmark Dataset")
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation="vertical")
ax.set_ylabel("CCR")
ax.set_ylim([0.995 * min(model_scores), max(model_scores) * 1.005])
colors = ["r"]*len(ovr_f1_score) + ["g"] * len(f1_score_temp)
ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
plt.tight_layout()
plt.show()