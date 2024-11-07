import pandas as pd
df=pd.read_csv(r'C:\Machine Learning\Train_Titanic.csv')
print(df)
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(data=df,x='Pclass',hue='Survived')
plt.show()
df['Age']=df['Age'].fillna(df['Age'].mean())
print(df)
from sklearn.preprocessing import LabelEncoder
Label = LabelEncoder()
df['Gender']=Label.fit_transform(df['Gender'])
print(df)
df=df.drop('Cabin',axis=1)
print(df)
x=df.drop(['Survived'],axis=1)
y=df['Survived']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print(x_test)
print(y_test)
print(x_train.size)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
print(y_predict)
print(y_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_predict,y_test))
