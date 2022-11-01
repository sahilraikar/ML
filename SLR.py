import pandas as pd
import matplotlib.pyplot as plt
# ypred = m*x + C
#
m=0
c=0
L=0.01
epoch =2000
data = pd.read_csv("Advertising.csv")
X= data["TV"]
Y = data["Sales"]
n = len(X)
for i in range(epoch):
 Ypred = m*X+c
 Dm=(-2/n)*sum(X*(Y-Ypred))
 Dn=(-2/n)*sum(Y-Ypred)
 m=m-L*Dm
 c=c-L*Dn
Y_pred = m*X + c
plt.scatter(X,Y)
plt.plot(X,Ypred,color="red")
plt.show()