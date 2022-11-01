import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

df=pd.read_csv('cars.csv')
le=LabelEncoder()
df["transmission"]=le.fit_transform(df["transmission"])
oe=OrdinalEncoder(categories=[['Test Drive Car','First Owner','Second Owner','Third Owner','Fourth & Above Owner']],dtype=int)
df[["owner"]]=oe.fit_transform(df[["owner"]])

x = df[['year_bought','km_driven','transmission','owner']]
y = df['selling_price']
x.insert(0,'x0',1)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=13)

xtrain=xtrain.values
ytrain=ytrain.values.reshape(len(ytrain),1)

theta=np.matmul(np.matmul(np.linalg.inv(np.matmul(xtrain.T,xtrain)),xtrain.T),ytrain)
print(theta)
ypred=np.matmul(xtest.values,theta)
print(ypred)

ax = plt.axes()
ax.scatter(range(len(ytest)),ytest)
ax.scatter(range(len(ytest)),ypred)
ax.ticklabel_format(style='plain')
plt.legend(['Actual','Predicted'])
plt.show()