import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
'''model train'''

y_pred = regressor.predict(X_test)
'''making prediction'''

# visualising the train set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs. Exprience (Train Set)')
plt.xlabel('Year of Exprience')
plt.ylabel('Salary')

# visualising the test set result
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
'''the shape of the line will not change when the model has already trained
  using X_train as input is just for a longer line'''
plt.title('Salary vs. Exprience (Test Set)')
plt.xlabel('Year of Exprience')
plt.ylabel('Salary')
