import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from scipy.io import arff
import pandas as pd

#This code tests the runtime of the Classifier Chain model against number of chains and number of labels

X, Y = fetch_openml("reuters", version=4, return_X_y=True)
Y = Y == "TRUE"
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


base_lr = LogisticRegression(max_iter=15000)
mydecision = DecisionTreeClassifier()
labelchains = [ClassifierChain(mydecision, order="random", random_state=i) for i in range((len(Y_train.columns)-1)*10)]
label_runtimes = []
numlabels = []

labelchains[0].fit(X_train, Y_train)
for labeltest in range(len(Y_train.columns)-1):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    label_chain = ClassifierChain(mydecision, order="random", random_state=labeltest)
    Y_label = Y_train
    
    Y_label.drop(Y_label.columns[2:7-labeltest], axis=1, inplace=True)
    Y_cur = Y_label
    
    labelchains[0].fit(X_train, Y_train)
    labelstart = datetime.datetime.now()
    label_chain.fit(X_train, Y_cur)
    labelend = datetime.datetime.now()
    
    label_runtimes.append((labelend - labelstart).total_seconds())
    numlabels.append(2+labeltest)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(Y_train.columns)        
print(label_runtimes)
print(numlabels)
x_pos_2 = np.arange(len(numlabels))
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.grid(True)
ax2.set_title("Classifier Chain Training Runtime vs Number of Labels")
ax2.set_xticks(x_pos_2)
ax2.set_xticklabels(numlabels, rotation="vertical")
ax2.set_ylabel("runtime (s)")
ax2.set_xlabel("number of labels")
ax2.set_ylim([min(label_runtimes) * 0.9, max(label_runtimes) * 1.1])
ax2.plot(x_pos_2, label_runtimes, alpha=0.5)

plt.tight_layout()
plt.show()


runtimes_decisiontree = []
runtimes_logistic = []
runtimes = []
number_of_chains_tracker = []
number_of_chains = 10
CCRSvsChains = []
JacvsChains = []
f1_scores = [[] for asdf in range(len(Y_train.columns))]
for runner in range(15):
    chains = [ClassifierChain(mydecision, order="random", random_state=i) for i in range(number_of_chains)]
    chainsstart = datetime.datetime.now()
    for chain in chains:
        chain.fit(X_train, Y_train)
    chainsend = datetime.datetime.now()
    runtimes_decisiontree.append((chainsend - chainsstart).total_seconds())
    runtimes.append((chainsend - chainsstart).total_seconds())
    chains = [ClassifierChain(base_lr, order="random", random_state=i) for i in range(number_of_chains)]
    chainsstart_log = datetime.datetime.now()
    for chain in chains:
        chain.fit(X_train, Y_train)
    chainsend_log = datetime.datetime.now()
    runtimes_logistic.append((chainsend_log - chainsstart_log).total_seconds())
    runtimes.append((chainsend_log - chainsstart_log).total_seconds())
    number_of_chains_tracker.append(number_of_chains)
    number_of_chains += 10
    Y_pred_chains = np.array([chain.predict(X_test) for chain in chains])
    chain_accuracy_scores = [
        accuracy_score(Y_test, Y_pred_chain >= 0.5)
        for Y_pred_chain in Y_pred_chains
    ]
    chain_jaccard_scores = [
        jaccard_score(Y_test, Y_pred_chain >= 0.5, average="samples")
        for Y_pred_chain in Y_pred_chains
    ]
    Y_pred_ensemble = Y_pred_chains.mean(axis=0)
    CCRSvsChains.append(accuracy_score(Y_test, Y_pred_ensemble >= 0.5))
    JacvsChains.append(jaccard_score(Y_test, Y_pred_ensemble >= 0.5, average="samples"))
    f1_score_temp = f1_score(Y_test, Y_pred_ensemble >= 0.5, average=None)
    for f1count in range(len(f1_score_temp)):
        f1_scores[f1count].append(f1_score_temp[f1count])


x_pos = np.arange(len(number_of_chains_tracker))
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.grid(True)
ax2.set_title("")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(number_of_chains_tracker, rotation="vertical")
ax2.set_ylabel("Accuracy")
ax2.set_xlabel("number of chains")
ax2.set_ylim(0.0, 1.1)
ax2.plot(x_pos, JacvsChains, alpha=0.5, color='r', label='Jaccard Score')
ax2.plot(x_pos, CCRSvsChains, alpha=0.5, color='b', label='CCR')
for f1count2 in range(len(f1_scores)):
    ax2.plot(x_pos, f1_scores[f1count2], alpha=0.5, label='F1 Score for Label ' + str(f1count2))
ax2.legend(loc='upper left')
plt.tight_layout()
plt.show()
## Plotting the runtime comparison between Decision Tree and Logistic Regression of CC
fig, ax = plt.subplots(figsize=(7, 4))
ax.grid(True)
ax.set_title("Classifier Chain Training Runtime")
ax.set_xticks(x_pos)
ax.set_xticklabels(number_of_chains_tracker, rotation="vertical")
ax.set_ylabel("runtime (s)")
ax.set_xlabel("number of chains")
ax.set_ylim([min(runtimes) * 0.9, max(runtimes) * 1.1])
ax.plot(x_pos, runtimes_decisiontree, alpha=0.5, color='r', label='Decision Tree')
ax.plot(x_pos, runtimes_logistic, alpha=0.5, color='b', label='Logistic Regression')
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()