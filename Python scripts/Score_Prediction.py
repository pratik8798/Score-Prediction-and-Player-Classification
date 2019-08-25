# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 08:16:16 2019

@author: Dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def mse(y_pred,y_test):
    sum=0.0
    
    for i in range(len(y_pred)):
        sum+=(y_test[i]-y_pred[i])**2
    
    return (sum/len(y_pred))


dataset=pd.read_excel("odidata.xlsx")

X=dataset.iloc[:,8:17].values
Y=dataset.iloc[:,17].values

features=["Runs","Wickets","Overs","Run rate","Runs last 5","Wickets Last 5","rr last 5","Striker","non striker"]

from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0,shuffle=True)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
x_test=sc_X.transform(x_test)

sc_Y=StandardScaler()
Y_train=sc_Y.fit_transform(Y_train.reshape(-1,1))
y_test=sc_Y.transform(y_test.reshape(-1,1))


from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(X_train,Y_train)

y_pred=regressor.predict(x_test)

test=np.array([[17,2,3.3,4.86,17,2,4.86,6,0]])
test=sc_X.transform(test)

y_t=regressor.predict(test)

y_t=sc_Y.inverse_transform(y_t)


y_test=sc_Y.inverse_transform(y_test)
y_pred=sc_Y.inverse_transform(y_pred)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_pred,y_test))





from sklearn.svm import SVR

regressor=SVR(kernel="rbf")
regressor.fit(X_train,Y_train)

y_pred=regressor.predict(x_test)



from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score

print(mean_absolute_error(y_test,y_pred))
mean_squared_error(y_test, y_pred)
print(regressor.score(x_test,y_test))


from sklearn.tree import DecisionTreeRegressor

regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,Y_train)

y_pred=regressor.predict(x_test)



from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score

print(mean_absolute_error(y_test,y_pred))
mean_squared_error(y_test, y_pred)


from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor(n_estimators=1000,random_state=0)
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(x_test)


#print(mse(y_test,y_pred))

from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score

print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test, y_pred))



#statistics part

from yellowbrick.regressor import ResidualsPlot
from sklearn.linear_model import Ridge

ridge=Ridge()
visualizer = ResidualsPlot(ridge, hist=False)
visualizer.fit(X_train, Y_train)
visualizer.score(x_test, y_test)
visualizer.poof()


from yellowbrick.features import Rank1D

visualizer=Rank1D(features=features,algorithm="shapiro")

visualizer.fit(X_train,Y_train)
visualizer.transform(X)

coef=regressor.coef_
