#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import svm


# In[48]:


df=pd.read_csv('C:/Users/digvi/Downloads/pima-indians-diabetes-database/diabetes.csv')
df.head()


# In[4]:


df.dtypes


# In[49]:


#float dtypes were converted to int dtypes
cdf=df[pd.to_numeric(df['BMI'],errors='coerce').notnull()]
cdf=df[pd.to_numeric(df['DiabetesPedigreeFunction'],errors='coerce').notnull()]
cdf['BMI'] = df['BMI'].astype('int')
cdf['DiabetesPedigreeFunction']=df['DiabetesPedigreeFunction'].astype('int')
cdf.dtypes

X=df[['Pregnancies','Glucose','BloodPressure','Insulin','SkinThickness','Age','BMI','DiabetesPedigreeFunction']].values
X=np.asarray(X)
X[0:5]
cdf['Outcome'] = cdf['Outcome'].astype('int')
y=np.asarray(cdf['Outcome'])
y[0:5]


# In[83]:


svc = sklearn.svm.SVC(kernel='rbf')
from sklearn.svm import SVC
svc = SVC()
num_obs = len(df)
num_true = len(df.loc[df['Outcome'] == 1])
num_false = len(df.loc[df['Outcome'] == 0])
print("Number of True cases:  {0} ({1:2.2f}%)".format(num_true, ((1.00 * num_true)/(1.0 * num_obs)) * 100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, (( 1.0 * num_false)/(1.0 * num_obs)) * 100))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.33,random_state=42)
print('Train set : ',X_train.shape, y_train.shape)
print('Test set: ',X_test.shape,y_test.shape)

#trai and test validation is done
trainval=(1.0 * len(X_train)) / (1.0 * len(df.index))
testval=(1.0*len(X_test))/(1.0*len(df.index))
print("{0:0.2f}% in training set".format(trainval * 100))
print("{0:0.2f}% in test set".format(testval * 100))

#to check the fit of training dataset
svc.fit(X_train,y_train)


# In[84]:


prediction_train_data=svc.predict(X_train)
prdiction_test_data=svc.predict(X_test)
from sklearn.metrics import jaccard_similarity_score


# In[85]:


yhat=svc.predict(X_test)


# In[86]:


jsc=jaccard_similarity_score(y_test,yhat)
print(jsc)
from sklearn.metrics import f1_score
print(f1_score(y_test, yhat, average='weighted')) 

