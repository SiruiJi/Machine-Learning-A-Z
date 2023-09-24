import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import data
dataset = pd.read_csv('/Social_Network_Ads.csv')

# indentify dependent and independent variable 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# take care of missing value
# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# feature scaling
'''feature scaling can be beneficial for logistic regression, especially in cases where 
regularization is applied or when using gradient descent for optimization.'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X_train = sc_X.fit_transform(X_train)
sc_X_test = sc_X.transform(X_test)

# bulid and train model
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state=0)
classifier.fit(sc_X_train, y_train)

# predict
print(classifier.predict(sc_X.transform([[30, 87000]])))

y_pred = classifier.predict(sc_X_test)
compare = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)),1)

# confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)

# visualising the result
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc_X.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc_X.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

'''predict(): This method returns the predicted class labels.
For binary classification, it will return either 0 or 1 (or whatever class labels you have).
It's essentially a thresholded version of .predict_proba(). 
By default, the threshold for binary classification is 0.5. 
If the predicted probability of the positive class is greater than or equal to 0.5, 
it predicts the positive class; otherwise, it predicts the negative class.

predict_proba(): This method returns the probabilities for each class.
For binary classification, it will return a two-dimensional array where each row corresponds to a sample and contains 
two values: the probability that the sample belongs to the negative class (first value) and the probability that the sample 
belongs to the positive class (second value). The sum of these two values will always be 1.For multiclass classification, 
it will return probabilities for each class, and the sum across all classes will be 1 for each sample.

predict_log_proba(): This method returns the logarithm of the probabilities obtained from .predict_proba().
The logarithm of probabilities can be useful in certain situations, especially when dealing with very small probability values, 
to prevent underflow in computations. Taking the logarithm can also sometimes lead to more stable numerical computations, 
especially in the context of multiplying probabilities (since multiplying probabilities corresponds to adding their logarithms).
Like .predict_proba(), for binary classification, it returns two values (log probabilities) for each sample, and for multiclass
classification, it returns log probabilities for each class for each sample.'''
