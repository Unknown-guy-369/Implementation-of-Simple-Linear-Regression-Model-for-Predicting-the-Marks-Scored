# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Abishek Priyan M
RegisterNumber:  212224240002
*/
```

```py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()



```

## Output:

![image](https://github.com/user-attachments/assets/19839221-616b-4473-8d3d-cdd036f013f3)

![image](https://github.com/user-attachments/assets/f0db926d-ba8e-41a2-b308-3e1664040e3f)

![image](https://github.com/user-attachments/assets/b31788ac-1864-4645-9d7f-96404ca174ed)

![image](https://github.com/user-attachments/assets/ea050c07-73ee-4d42-82bb-d319b2552191)

![image](https://github.com/user-attachments/assets/e5550ff8-892e-432e-929a-95e05d5d4cd2)

![image](https://github.com/user-attachments/assets/e8475ddb-3aa3-4d6b-8884-cd1ad7f26c42)

![image](https://github.com/user-attachments/assets/13accdec-c895-442d-ae33-da8381c0e5ee)

![image](https://github.com/user-attachments/assets/af1575f3-7e52-4991-a620-a668dca76842)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
