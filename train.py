import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()
model = BaggingClassifier(
DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=30),
n_estimators=750, n_jobs=-1, random_state=42).fit(X, y)


with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
