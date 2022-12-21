#!/usr/bin/env python
# coding: utf-8

# In[44]:


#Using Classification model for testing the accuracy of inputs vs ouputs
#importing the dataset
import pandas as pd
ds=pd.read_csv('diabetes.csv')


# In[46]:


#dividing into inputs and outputs
array=ds.values
x=array[:,0:8]
y=array[:,8]


# In[47]:


#divide the ds into traning and testing 
from sklearn.model_selection import train_test_split

seed=50
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=seed)


# built model
from sklearn.ensemble import RandomForestClassifier
rd=RandomForestClassifier(n_estimators = 100)
#model training
rd.fit(x_train,y_train)


# In[48]:


#testing
predict=rd.predict(x_test)


# In[51]:


#performance
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predict)
accuracy
print('the percentage of accuracy:',accuracy*100)


# In[52]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predict)
cm


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm,annot=True)
plt.show()


# In[27]:




