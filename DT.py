import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz

df=pd.read_csv('heart.csv')
l=len(df.columns)
x=df.iloc[:,:l-1]
y=df['target']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=13)
cgini=DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_leaf=5,random_state=13)
cgini.fit(xtrain,ytrain)
ypred=cgini.predict(xtest)
print(accuracy_score(ytest,ypred))
feature_names = ["age", "sex"," cp ", " trestbps"," chol"," fbs" ,"restecg" ," thalach " ,"exang" , " oldpeak" ,"slope" ," ca", " thal"]
target_names = ['HD-Yes', 'HD-No']
dot_data = tree.export_graphviz(cgini,out_file=None,
feature_names=feature_names,class_names=target_names, filled=True)
# Draw tree
tr = graphviz.Source(dot_data, format="png")
print(tr)
