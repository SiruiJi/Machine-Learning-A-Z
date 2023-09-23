import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import the data
dataset = pd.read_csv('/Position_Salaries.csv')

# indentify the dependent and independent variable 
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# taking care of the missing data
# train test split
# feature scaling 
''' For decision tree or random forest regression model, the results are from successive splits of tje data,
through the different nodes of the tree. Therefore there are no equations like the previous models. 
And that why no feature scaling is needed to.'''

# build and train the model 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# predict
print(regressor.predict([[6.5]]))

# visualising the outcome 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
