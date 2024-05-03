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

# Load a multi-label dataset from https://www.openml.org/d/40597
#data, meta = arff.loadarff('Yelp.arff')
#df = pd.DataFrame(data)
#df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
#features = df.drop(columns=['IsRatingBad', 'IsRatingModerate', 'IsRatingGood',
#                   'IsFoodGood', 'IsServiceGood', 'IsAmbianceGood', 'IsDealsGood', 'IsPriceGood'])
#labels = df[['IsFoodGood', 'IsServiceGood', 'IsAmbianceGood',
#             'IsDealsGood', 'IsPriceGood']].astype(int)
#X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=0)


# Load a multi-label dataset from https://www.openml.org/d/40597
#X, Y = fetch_openml("scene", version=4, return_X_y=True)
#Y = Y == "TRUE"
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train =  pd.read_csv('cc_x_train.csv')
Y_train = pd.read_csv('cc_y_train.csv')
X_test =  pd.read_csv('cc_x_test.csv')
Y_test = pd.read_csv('cc_y_test.csv')
base_lr = LogisticRegression(max_iter=15000)
mydecision = DecisionTreeClassifier()
ovr = OneVsRestClassifier(mydecision)
overtimestart = datetime.datetime.now()
ovr.fit(X_train, Y_train)
ovrtimeend = datetime.datetime.now()
print("OVR Time: ", ovrtimeend - overtimestart)
Y_pred_ovr = ovr.predict(X_test)
ovr_jaccard_score = jaccard_score(Y_test, Y_pred_ovr >= 0.5, average="samples")
chains = [ClassifierChain(mydecision, order="random", random_state=i) for i in range(10)]
chainsstart = datetime.datetime.now()
#presetchain = ClassifierChain(mydecision, order=[1,0,2,6,3,4,5]) #[5,4,3,6,2,0,1]

#presetchain.fit(X_train, Y_train)
#Y_pred_preset = presetchain.predict(X_test)
#preset_jaccard_score = jaccard_score(Y_test, Y_pred_preset >= 0.5, average="samples")
for chain in chains:
    chain.fit(X_train, Y_train)
chainsend = datetime.datetime.now()
print("Chains Time: ", chainsend - chainsstart)
Y_pred_chains = np.array([chain.predict(X_test) for chain in chains])
chain_jaccard_scores = [
    jaccard_score(Y_test, Y_pred_chain >= 0.5, average="samples")
    for Y_pred_chain in Y_pred_chains
]

#Y_pred_ensemble = np.median(Y_pred_chains,axis=0)
Y_pred_ensemble = Y_pred_chains.mean(axis=0)
ensemble_jaccard_score = jaccard_score(
    Y_test, Y_pred_ensemble >= 0.5, average="samples"
)

model_scores = [ovr_jaccard_score] + chain_jaccard_scores + [ensemble_jaccard_score]

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
ax.set_title("Classifier Chain Ensemble Performance Comparison - Decision Tree - PC Dataset")
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation="vertical")
ax.set_ylabel("Jaccard Similarity Score")
ax.set_ylim([min(model_scores) * 0.95, max(model_scores) * 1.05])
colors = ["r"] + ["b"] * len(chain_jaccard_scores) + ["g"]
ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
plt.tight_layout()
plt.show()