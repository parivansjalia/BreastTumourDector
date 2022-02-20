#Description: This program detect and classify breast cancer based off of data.
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load the data
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('data.csv')
df.head(7)

#Count the number of rowss and columns in the data set
df.shape

#Count the number of empty (NaN, NAN, na) values in each column
df.isna().sum()

#drop the last column (it's empty)
df = df.dropna(axis=1)

#Get the new count of the data
df.shape

#Get a count of the number of Malignangt (M) or Benign (B) cells
df['diagnosis'].value_counts()

#Visualize the count
sns.countplot(df['diagnosis'], label='count')

#Look at the data types to see which colums need to be encoded (transform into a number value)
df.dtypes

#encode the cateforical data values (M and B)
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)

#create a pair plot
sns.pairplot(df.iloc[:,1:5], hue='diagnosis')

#print the first 5 rows of the new data
df.head(5)

#get the correlation of the colums
df.iloc[:,1:12].corr()

#visulize the correlation
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:12].corr(), annot=True, fmt = ".0%")

#Spilt the data set into x and y 
X = df.iloc[:,2:31].values
Y = df.iloc[:,1].values

#split the dataset into 75% training and 25% testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25 , random_state = 0)

#scale the data (feature scaling) 0 or 1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#create a function for the models
def models(X_train, Y_train):

  #Logistic Regression
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state=0)
  log.fit(X_train, Y_train)

  #Descision Tree
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
  tree.fit(X_train, Y_train)

  #Random Forest Classifer
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = 0)
  forest.fit(X_train, Y_train)

  #Print the models accruary on the training data
  print('[0]LogisticRegression Traing Accuracy:', log.score(X_train, Y_train))
  print('[1]Decision Tree Classifer Accuracy:', tree.score(X_train, Y_train))
  print('[2]Random Forest Classifier Traning Accuracy', forest.score(X_train, Y_train))

  return log, tree, forest

#Getting all of the models
model = models(X_train, Y_train)

#test model accurary on test data on consusion matrix (test accurary)
from sklearn.metrics import confusion_matrix

for i in range(len(model)):
  print('Model ', i)
  cm = confusion_matrix(Y_test, model[i].predict(X_test))

  TP = cm[0][0]
  TN = cm[1][1]
  FN = cm[1][0]
  FP = cm[0][1]

  print(cm)
  print('Testing Accuracy = ', (TP + TN)/(TP + TN + FN + FP))

  
 #Show another way to get metrics of the models
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
  print('Model ', i)
  print()
  print( classification_report(Y_test, model[i].predict(X_test)))
  print()
  print( accuracy_score(Y_test, model[i].predict(X_test)))
  
  #Print the prediction of Random Forest Calssifier Model
pred = model[2].predict(X_test)
print(pred)
print()
print(Y_test)

#train through xgboost

import xgboost as xgb

boost = xgb.XGBClassifier(silent = True, 
                      scale_pos_weight = 1,
                      learning_rate = 1,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective = 'binary:logistic', 
                      n_estimators = 10000, 
                      reg_alpha = 0.3,
                      max_depth = 4, 
                      min_child_weight = 1,
                      eta = 0.1,
                      gamma = 1)
boost.fit(X_train, Y_train)

print('[3]XGBoost Traning Accuracy', boost.score(X_train, Y_train))

#test through xgboost

print( classification_report(Y_test, boost.predict(X_test)))
print()
print( accuracy_score(Y_test, boost.predict(X_test)))

#prediction for xgboost
preed = boost.predict(X_test)
print(preed)
print()
print(Y_test)
