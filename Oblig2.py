from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets.squeeze()  # Ensure y is 1D

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


# 1) set up grid of hyper‐parameters
param_grid = {
    'n_estimators': [10, 50, 100, 150, 200],
    'max_depth': [5, 50, None],
    'max_features': [2, 3, None]
}
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

gs_reg = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)

gs_reg.fit(X_train,y_train)

y_pred = gs_reg.predict(X_test)

print(cross_val_score(y_pred,y_test))
