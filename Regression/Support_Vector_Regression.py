import pandas as pd
import matplotlib.pyplot as plt


# import data
dataset = pd.read_csv('/Position_Salaries.csv')

# indentify the dependent and independent variables
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y  = y.reshape(len(y),1)
'''to reshape the array y into a two-dimensional array with as many rows as y has elements and 1 column.'''

# taking care of the missing data
# train test split

# feature scaling
''' We are going to apply feature scaling for this model, because, in the SVR model
there is no explicit equation of the dependent with respect to the features, and there are no coefficients 
multiplying each of the features, and therefore, no compensating with lower values for the features taking high values.
SVR model has a implicit equation of the dependent variable with respect to these features, we don't have such coefficient.'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
'''two separate instances of StandardScaler are to ensure that the scaling parameters (mean and standard deviation) learned 
from the input features X are not mixed with those from the target y.Using two separate instances is a safer approach that 
provides more flexibility, especially when X and y have different distributions.'''
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# build and train the model
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# predict
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1)))

# visualizing the result
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color='blue')
