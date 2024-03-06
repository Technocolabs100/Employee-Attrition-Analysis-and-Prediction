#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd


# In[27]:


import numpy as np


# In[28]:


df = pd.read_csv('/home/hp/Employee-Attrition-Analysis-and-Prediction/Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[29]:


df.head()


# In[30]:


df.tail()


# In[31]:


df.shape


# In[32]:


df.describe()


# In[33]:


df.columns


# In[34]:


df.info()


# In[35]:


df.isnull().sum() 


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


# pip install scikit-learn


# In[54]:


X = df.drop(columns=['Attrition'])


# In[55]:


y = df['Attrition']


# In[63]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[64]:


x_train


# In[65]:


from sklearn.compose import ColumnTransformer


# In[71]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# In[78]:


transformer = ColumnTransformer(
    transformers=[
        ('ohc_tnf', OneHotEncoder(sparse_output=False, drop='first'), ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']),
    ],remainder='passthrough'
)


# In[80]:


x_train_transformed = transformer.fit_transform(x_train)


# In[81]:


x_train_transformed


# In[82]:


x_test_transformed = transformer.fit_transform(x_test)


# In[83]:


x_test_transformed


# In[84]:


y_train


# In[88]:


le = LabelEncoder()


# In[91]:


le.fit(y_train)


# In[92]:


y_train_transformed = le.transform(y_train)


# In[93]:


y_train_transformed


# In[94]:


le.fit(y_test)


# In[95]:


y_test_transformed = le.transform(y_test)


# In[ ]:


y_test_transformed


