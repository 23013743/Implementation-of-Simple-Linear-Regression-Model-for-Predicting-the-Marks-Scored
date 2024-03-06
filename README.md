# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NIRMAL.N
RegisterNumber:  212223240107
*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

# Dataset
![WhatsApp Image 2024-03-06 at 05 10 46_5d68cd03](https://github.com/23013743/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/161271714/d250b899-c657-472e-bbd9-211b9eab8ac2)

# Headvalue

![WhatsApp Image 2024-03-06 at 05 11 37_65eef1f1](https://github.com/23013743/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/161271714/3493e532-2af2-4d13-824e-c4d8e3add345)

# Tailvaule

![WhatsApp Image 2024-03-06 at 05 12 54_ddad704e](https://github.com/23013743/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/161271714/64bac334-751b-44b1-ac2e-af06004dfe5f)

# X and Y Vaule

![WhatsApp Image 2024-03-06 at 05 14 19_d591d647](https://github.com/23013743/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/161271714/10d1b763-fc45-420a-8a66-c680937fc7be)

# Predication vaule of X and Y

![WhatsApp Image 2024-03-06 at 05 15 13_cd2807df](https://github.com/23013743/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/161271714/e1360751-6fba-48f8-8ab7-32226bdecee7)

# MSE,MAE and RMSE

![WhatsApp Image 2024-03-06 at 05 16 15_4d145bb8](https://github.com/23013743/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/161271714/b1481fa4-ed59-429e-951b-88ada7d68954)

# Tranning set

![WhatsApp Image 2024-03-06 at 05 17 57_efda330a](https://github.com/23013743/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/161271714/cd399160-fd33-4c41-919d-0975d79d4f01)

# Testing set

![WhatsApp Image 2024-03-06 at 05 19 14_7893eea6](https://github.com/23013743/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/161271714/a7cc525c-0831-43b2-886a-9c50f975aca9)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
