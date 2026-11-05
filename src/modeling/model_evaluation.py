import numpy as np
import pandas as pd

import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

positive_label = "happiness"


clf = pickle.load(open('model.pkl','rb'))
test_data = pd.read_csv('./data/features/test_bow.csv')

X_test = test_data.iloc[:,0:-1].values
y_test = test_data.iloc[:,-1].values

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
# Accuracy does not take average or zero_division.
accuracy = accuracy_score(y_test, y_pred)

# Macro averaging evaluates both emotions with equal weight.
precision = precision_score(
    y_test, y_pred, average="macro", zero_division=0
)

recall = recall_score(
    y_test, y_pred, average="macro", zero_division=0
)

f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'f1':f1
}

with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)