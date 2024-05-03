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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import pandas as pd


###X = np.array(ct.fit_transform(X), dtype = str)


#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Y_train = pd.read_csv('binary_final_labels.csv')


mycorrelations = pd.read_csv('cc_correlations.csv')

X_train =  pd.read_csv('cc_x_train.csv')

Y_train = pd.read_csv('cc_y_train.csv')
print(Y_train.corr())
#print(Y_train[:][Y_train.columns[0]])
print(mycorrelations.corr())


#X_test =  pd.read_csv('full_done_feat_test.csv')
# Load a multi-label dataset from https://www.openml.org/d/40597
#X, Y = fetch_openml("yeast", version=4, return_X_y=True)
#Y = Y == "TRUE"
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#X, Y = make_multilabel_classification(n_classes=20, n_labels=2,
#ohe=OneHotEncoder()
#ct=ColumnTransformer(transformers=[('encoder',ohe,[0])],remainder='passthrough')
#print(X)
#print(Y)
#X_train=np.array(ct.fit_transform(X))
#Y_train=np.array(ct.fit_transform(Y))
#print(X_train)
#X_test = pd.read_csv('CC_TEST.csv')
mylogistic = LogisticRegression(max_iter=15000)
mysvc = SVC()
ovr = OneVsRestClassifier(mysvc)
myknr = KNeighborsRegressor()
presetchain = ClassifierChain(mysvc, order=None)
presetchain.fit(X_train, Y_train)
ovr.fit(X_train, Y_train)
Y_pred_ovr = ovr.predict(X_train)

ovr_jaccard_score = accuracy_score(Y_train, Y_pred_ovr)
Y_predict_preset = np.array(presetchain.predict(X_train))
chains = [ClassifierChain(mylogistic, order="random", random_state=i) for i in range(6)+42]
for chain in chains:
    chain.fit(X_train, Y_train)

Y_pred_chains = np.array([chain.predict(X_train) for chain in chains])
print(Y_pred_chains)

f1_scores_temp = []
f1_scores_fin = []
f1x = 0
for chain in chains:
    f1y = 0
    f1_scores_temp = []
    for label in Y_train.columns:
        y_pred = chain.predict(X_train)
       
        f1_scores_temp.append(f1_score(Y_train, y_pred, average='macro'))
    f1_scores_fin.append(f1_scores_temp)
    f1x = f1x+1
    
print(f1_scores_fin)
chain_jaccard_scores = [
    accuracy_score(Y_train, Y_pred_chain >= 0.5)
    for Y_pred_chain in Y_pred_chains
]

accuracy_preset = accuracy_score(Y_train, Y_predict_preset >= 0.5)
Y_pred_ensemble = Y_pred_chains.mean(axis=0)
ensemble_jaccard_score = accuracy_score(Y_train, Y_pred_ensemble >= 0.5)
model_scores = [ovr_jaccard_score] + chain_jaccard_scores + [ensemble_jaccard_score]

model_names = (
    "OVR",
    "Chain 1",
    "Chain 2",
    "Chain 3",
    "Chain 4",
    "Chain 5",
    "Chain 6",
    "Ensemble",
)

x_pos = np.arange(len(model_names))

fig, ax = plt.subplots(figsize=(7, 4))
ax.grid(True)
ax.set_title("Classifier Chain Order Performance Comparison")
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation="vertical")
ax.set_ylabel("CCR")
ax.set_ylim([min(model_scores) * 0.9, max(model_scores) * 1.1])
colors = ["b"] * len(chain_jaccard_scores) + ["r"] + ["g"]
ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
plt.tight_layout()
plt.show()