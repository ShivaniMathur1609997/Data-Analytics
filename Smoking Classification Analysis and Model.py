#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('smoking.csv')


# In[3]:


df


# In[4]:


df.info()


# In[5]:


from pandas_profiling import ProfileReport


# In[6]:


profile = ProfileReport(df)
profile.to_notebook_iframe()


# In[7]:


from pycaret.classification import *


# In[8]:


s = setup(df,target ='smoking',session_id = 10)


# In[9]:


best = compare_models()


# In[10]:


df.info()


# In[11]:


df['gender'] = df['gender'].map({'M':0,'F':1})


# In[12]:


df['gender']


# In[13]:


df['tartar'].unique()


# In[14]:


df['tartar'] = df['tartar'].map({'Y':0,'N':1})


# In[15]:


df['tartar']


# In[16]:


df.info()


# In[18]:


df.drop('oral',axis=1,inplace=True)


# In[19]:


df


# In[20]:


df.columns


# In[21]:


df.info()


# In[22]:


X = df.iloc[:,:-1]


# In[23]:


Y = df.iloc[:,-1]


# In[24]:


X


# In[25]:


Y


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.33,random_state=42)


# In[28]:


x_train


# In[29]:


x_test


# In[30]:


y_train


# In[31]:


y_test


# In[32]:


from sklearn.ensemble import RandomForestClassifier


# In[34]:


model = RandomForestClassifier()


# In[35]:


model.fit(x_train,y_train)


# In[36]:


y_pred = model.predict(x_test)


# In[37]:


y_pred


# In[38]:


from sklearn.metrics import accuracy_score, classification_report


# In[39]:


accuracy_score(y_pred,y_test)


# In[41]:


print(classification_report(y_pred,y_test))


# In[42]:


print('accuracy-Random Forest Classifier',accuracy_score)


# In[43]:


from sklearn.linear_model import LogisticRegression


# In[44]:


model2 = LogisticRegression()


# In[51]:


model2.fit(x_train,y_train)


# In[52]:


y2_pred = model2.predict(x_test)


# In[53]:


y2_pred


# In[54]:


from sklearn.metrics import accuracy_score, classification_report


# In[57]:


accuracy2 = accuracy_score(y2_pred,y_test)


# In[59]:


accuracy2


# In[69]:


print('Accuracy score for Logistic Regression:',accuracy2)


# In[79]:


print(classification_report(y_test,y2_pred))


# In[63]:


from sklearn.tree import DecisionTreeRegressor


# In[64]:


model3 = DecisionTreeRegressor()


# In[65]:


model3.fit(x_train,y_train)


# In[66]:


y3_pred = model3.predict(x_test)


# In[71]:


y3_pred


# In[72]:


accuracy3 = accuracy_score(y3_pred,y_test)


# In[74]:


print('Accuracy score for Decision Tree Regressor:',accuracy3)


# In[76]:


print(classification_report(y3_pred,y_test))


# In[ ]:




