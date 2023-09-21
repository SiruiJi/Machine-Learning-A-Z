import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the data
dataset = pd.read_csv('/Position_Salaries.csv')

# indentify the independent and dependent variable
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]

# taking care of the missing value
# encoding catagorical varibale
# train_test_split
# feature scaling

# build and train the model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
'''convert the independent variable into polynomial format'''
regressor = LinearRegression()
regressor.fit(X_poly, y)
'''still apply linear model to build and train'''

# visualize the outcome
X_grid = np.arange(min(X.values), max(X.values), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(poly_reg.fit_transform(X_grid)), color='blue')

# apply the model
y_pred = regressor.predict(poly_reg.fit_transform([[6.5]]))
print(y_pred)
                       
