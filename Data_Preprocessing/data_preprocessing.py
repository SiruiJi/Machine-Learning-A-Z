# load the raw data and identify the dependent and independent variables
import pandas as pd
import numpy as np

dataset = pd.read_csv('D:/PyCharm Community Edition 2023.1.2/Python_Project/Professional/Machine Learning A-Z/Data.csv') # restore the uploaded file, and remember to add a '/' in front of the file name
X = pd.DataFrame(dataset.iloc[:, :-1])
''' iloc stand for index of column, '[]' used to select the rows and columns, 
 ':' means a range, and without neither lower nor upper bound, we select all the rows, after ',' we take care of the column,
  still ':' stand for range, the upper bound -1 means we take every column but the last one '''

y = pd.DataFrame(dataset.iloc[:, -1])
'''since we want only one column, we don't want a range, so no ':' after comma, '-1' is the index of the last column'''

# taking care of the missing value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
'''import this function to handle missing value by replace with the average value'''
imputer.fit(X.iloc[:, 1:])
X.iloc[:, 1:] = imputer.transform(X.iloc[:, 1:])
'''apply this method with only numerical variables'''

# transfer string into the dummy variables / encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [0])], remainder='passthrough')
X = ct.fit_transform(X)
feature_names = ct.get_feature_names_out()
X = pd.DataFrame(X, columns=feature_names)
print(X)

# encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
y = pd.DataFrame(y)
print(y)

# data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
'''feature scaling is a technique that will get the mean and the standard deviation of feature, 
    so you have to scaling the feature after the train test split'''

# feature scaling
''' For certain machine learning models, to avoid some features dominated other features, we apply feature scaling'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train.iloc[:, 3:] = sc.fit_transform(X_train.iloc[:, 3:])
'''fit: This computes the mean and standard deviation of the features in X_train. 
  transform: This scales the features of X_train using the computed mean and standard deviation.'''

X_test.iloc[:, 3:] = sc.transform(X_test.iloc[:, 3:])
'''transform: This scales the features of X_test using the mean and standard deviation that were computed from X_train.
DON'T scales the dummy variables. '''

print(X_train)
print(X_test)
