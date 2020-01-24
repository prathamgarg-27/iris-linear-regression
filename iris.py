
from sklearn.linear_model import LinearRegression # Algorithm / model
from sklearn.datasets import load_iris 
from sklearn.metrics import r2_score, mean_squared_error # evaluation metrices
from sklearn.model_selection import train_test_split # to split data into training and testing
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#%matplotlib inline

def myModel(features, target):
    """
        myModel will return m,c and error for each iteration
    """
    X_train, X_test, y_train, y_test = train_test_split(features, target,test_size=0.25)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_actual = y_test
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    rmse = np.sqrt(mse)
    return model.coef_, model.intercept_, rmse, r2, model

results = []

iris = load_iris()
features = iris['data']
target = iris['target']

for var in range(100):
    m,c,err,acc, model = myModel(features, target)
    l = [m,c,err,acc,model]
    results.append(l)

results = np.array(results)    
error = results[:, 2]

count=1
for i in range(0,99):
    if count > error[i]:
        count=error[i]
    else:
        count=count
for i in range(0,99):
    if error[i]==count:
        n=i    
m=results[:,0][n]
c=results[:,1][n]
model = results[:,-1][n]

def irismodel():
    return model