from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = fetch_california_housing()

dataset = pd.DataFrame(df.data)

dataset.columns = df.feature_names


# Independent and Dependent Features

x = dataset
y = df.target


# Train Test Split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.38, random_state=42)

# Standardizing the Dataset

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

regression = LinearRegression()
regression.fit(x_train,y_train)

# Cross Validation
mse = cross_val_score(regression, x_train, y_train, scoring='neg_mean_squared_error', cv=5)
np.mean(mse)

# Prediction

ref_predict = regression.predict(x_test)

sns.displot(ref_predict-y_test, kind="kde")
plt.show()

#adjusted r_square
score = r2_score(ref_predict, y_test)
print(score)