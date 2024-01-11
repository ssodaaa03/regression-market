import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

file = 'market_data.csv'

data = pd.read_csv(file).dropna()
train = data[:900]

x = train.drop(['Sale'], axis = 1)
y = train['Sale']
x_test = data.drop(['Sale'], axis = 1)[900:]
y_test = data['Sale'][900:]

model = LinearRegression()

model.fit(x, y)

y_pred = model.predict(data.drop(['Sale'], axis = 1)[900:])

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)