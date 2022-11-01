import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
df=pd.read_csv('Social_Network_Ads.csv')
x=df.iloc[:,2:4]
y=df["Purchased"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=13)
clf=SVC(kernel='linear')
clf.fit(xtrain,ytrain)
ypred=clf.predict(xtest)
cnf=confusion_matrix(ytest,ypred)
score=accuracy_score(ytest,ypred)
print(cnf)
print(score*100)
print(classification_report(ytest,ypred))
x1=df['Age']
x2=df['EstimatedSalary']
cnt=0
for i in df['Purchased']:
 if int(i)==0:
  plt.plot([x1[cnt]],[x2[cnt]],marker='o',c="red")
 else:
  plt.plot([x1[cnt]], [x2[cnt]], marker='o',c="green")
  cnt+=1
w=clf.coef_[0]
b=clf.intercept_[0]
xp=np.linspace(-1,60)
yp=-(w[0]/w[1])*xp-b/w[1]
plt.plot(xp,yp,c="blue")
plt.show()