import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df=pd.read_csv('BankNote_Authentication.csv')
l=len(df.columns)
x = df.iloc[:,:l-3]
y=df['class']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=100)
lr=LogisticRegression(random_state=100)
lr.fit(x,y)
ypred=lr.predict(xtest)
cnf=metrics.confusion_matrix(ytest,ypred)
print(cnf)