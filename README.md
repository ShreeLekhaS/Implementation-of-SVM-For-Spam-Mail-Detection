# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the Program.
2. Import the necessary packages.
3. Read the given csv file and display the few contents of the data.
4. Assign the features for x and y respectively.
5. Split the x and y sets into train and test sets.
6. Convert the Alphabetical data to numeric using CountVectorizer.
7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8. Find the accuracy of the model.
9. End the Program.

## Program and Output:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Shree Lekha.S
RegisterNumber: 212223110052
*/
```
```
import pandas as pd

data=pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()

```
![image](https://github.com/user-attachments/assets/ea250a90-33c5-4215-ab33-1d31f9f2c9f2)

```
data.info()
```
![image](https://github.com/user-attachments/assets/a4319b09-859b-481e-bb1f-9f03a0f0399b)

```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/fde3dd7c-c564-4bf6-a256-3094df91ca3e)

```
x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()

svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

```
![image](https://github.com/user-attachments/assets/08b6b9e9-519b-4692-8f9e-54b334fbae31)

```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

![image](https://github.com/user-attachments/assets/2afe8f11-c832-4208-8921-258723d9c087)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
