import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

file = 'market_data.csv'

data = pd.read_csv(file).dropna()
train = data[:900]

x = train.drop(['Sale'], axis = 1)
y = train['Sale']
x_test = data.drop(['Sale'], axis = 1)[900:]
y_test = data['Sale'][900:]

corr_matrix = train.corr()
target = corr_matrix['Sale']
r = target[abs(target) < 0.5].index.tolist()

x_train = x.drop(r, axis = 1)

model = LinearRegression()

model.fit(x_train, y)

y_pred = model.predict(x_test.drop(r, axis = 1))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)