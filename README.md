# Machine-Learning-Project
"first we use some python libraries"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"download transactiondata.csv from kaggle"
"now we read this data"
data=pd.read_csv("transactiondata.csv")
data.head()

"now we check for null entries in .csv"
print(data.isnull().sum())

"now we create a pie chart for our data"
import plotly.express as px
figure = px.pie(data, 
             values=quantity, 
             names=transaction,hole = 0.3, 
             title="Distribution of Transaction Type")
figure.show()

"we must have to check the dependency of column on each other"
correlation=data.corr()
correlation.head()

"draw a heatmap to find the null entries in .csv"
import seaborn as sns
sns.heatmap(data.isnull(),cmap='viridis')

"in data there is chash_out,Payment ... are must be convert in integer"
data["type"]=data["type"].map({"CASH_OUT":1,"PAYMENT":2,"CASH_IN":3,"TRANSFER":4,"DEBIT":5})
data["isFraud"]=data["isFraud"].map({0:"No Fraud",1:"Fraud"})
data.head()

"Train test split is a model validation procedure that allows you to simulate how a model would perform on new/unseen data"
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=100)

"DecisionTreeClassifier is used to predict for new entries"
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

"this tell about our model accuracy"
print(model.score(xtest, ytest))

"now we check"
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))
