import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import data
dataset = pd.read_csv('/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# taking care of missing value 
# train test split
# feature scaling

# bulid and train the model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=10)
regressor.fit(X,y)

# predict
print(regressor.predict([[6.5]]))

# visualising the outcome 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
