import pandas as pd

# import data
dataset = pd.read_csv('/50_Startups.csv')

# identify dependent and independent variable
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# taking care of the missing data

# encoding the categorical variable 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [3])], remainder='passthrough')
X = ct.fit_transform(X)
feature_names = ct.get_feature_names_out()
X = pd.DataFrame(X, columns=feature_names)

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling 
'''In mulitple linear regression, we actually don't need to apply feature scaling, because accroding to the function of the 
multiple linear regression, we have the coefficients that multipled to each independent variable, therefore, it doesn't matter
if some feature have higher values than other.'''

'''We don't need to worry about the select the features that have the highest P-value. Because Multiple Linear Regression model
will automatically identify the best features, the features that have the highest P values or that are the most statistically 
significant to figure out how to predict the dependent variable.'''

# model train 
from sklearn.linear_model import LinearRegression
regressior = LinearRegression()
regressior.fit(X_train, y_train)

y_pred = regressior.predict(X_test)

# model evaluate 
print(pd.DataFrame([y_pred, y_test]))

'''We don't need check the linear assumption before we apply dataset to linear model. We can just evaluate the model preform.
if its proform poorly, we just switch to other models.'''
