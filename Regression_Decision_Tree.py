import pandas as pd
import numpy as np
import matplotlib as plt

data=pd.read_csv('C:/Users/Bachi/Desktop/petrol_consumption.csv')
#print(data)
#print(data.describe())
#print(data.columns)

x=data.drop('Petrol_Consumption',axis=1)
y=data['Petrol_Consumption']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor()
reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)


df=pd.DataFrame({'Actual':y_test,  'Predicted':y_pred})
print(df)


from sklearn import metrics
print("mean_absolute_error :",metrics.mean_absolute_error(y_test,y_pred))
print("mean_squared_error :",metrics.mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error :",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))