from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ridge_regressor = Ridge()

df = fetch_california_housing()

dataset = pd.DataFrame(df.data)

dataset.columns = df.feature_names


# Independent and Dependent Features

x = dataset
y = df.target


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.38, random_state=42)

# Standardizing the Dataset

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

parameters = {"alpha": [1,2,5,10,30,48,59,98]}
ridgecv = GridSearchCV(ridge_regressor, parameters, scoring="neg_mean_squared_error", cv=5)
print(ridgecv.fit(X_train, y_train))
print(ridgecv.best_params_)
print(ridgecv.best_score_)

ridge_pred = ridgecv.predict(X_test)
print(sns.displot(ridge_pred-y_test, kind="kde"))

score = r2_score(ridge_pred, y_test)
print(score)

# --- LASSO Regression --- #

lasso = Lasso()

parameters = {"alpha": [1,2,5,10,30,48,59,98]}
lassocv = GridSearchCV(lasso, parameters, scoring="neg_mean_squared_error", cv=5)

lassocv.fit(X_train, y_train)
print(lassocv.best_params_)
print(lassocv.best_score_)

lasso_pred = lassocv.predict(X_test)
print(sns.displot(lasso_pred-y_test, kind="kde"))