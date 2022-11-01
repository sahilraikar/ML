import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv("titanic.csv")
df=df.dropna(axis=0,how="any",inplace=False)
inputs=df[["Pclass","Age","Fare"]].copy()
target=df.Survived
x1=inputs
y1=target
xtrain,xtest,ytrain,ytest=train_test_split(x1,y1,test_size=0.25,random_state=14)
lr=LogisticRegression()
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)
a1=accuracy_score(ytest,ypred)
scaler=StandardScaler()
scaler.fit(inputs)
finput=scaler.transform(inputs)
print(finput)
pca=PCA(n_components=2)
pca.fit(finput)
x=pca.transform(finput)
print(pca.components_)
print(pca.explained_variance_)
print(x)
x1=x
y1=target
xtrain,xtest,ytrain,ytest=train_test_split(x1,y1,test_size=0.25,random_state=14)
lr=LogisticRegression()
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)
a2=accuracy_score(ytest,ypred)
print(a1*100)
print(a2*100)