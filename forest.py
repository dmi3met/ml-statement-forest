# -*- coding: utf-8 -*-

import pandas as pd
import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

data = pd.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

kf = KFold(n_splits=5, random_state=1, shuffle=True)

for n_esti in range(1,51):
    clf = RandomForestRegressor(n_estimators=n_esti, random_state=1)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        score = r2_score(y_test, clf.predict(X_test))
        if score>0.52:
            print(n_esti, score)