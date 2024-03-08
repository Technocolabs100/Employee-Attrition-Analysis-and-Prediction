#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head() #Display first five rows


# In[2]:


df.tail()


# In[3]:


df.shape


# In[4]:


df.describe() # Display summary statistics in-terms of count,mean,std,min,max and percentages


# In[5]:


df.columns


# In[6]:


df.info() #Display information about the DataFrame


# In[7]:


df.isnull().sum() # Display missing_values_count per column


# In[8]:


from sklearn.model_selection import train_test_split
X = df.drop(columns=['Attrition'])
y = df['Attrition']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[9]:


x_train


# In[10]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# In[11]:


transformer = ColumnTransformer(
    transformers=[
        ('ohc_tnf', OneHotEncoder(sparse_output=False, drop='first'), ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']),
    ],remainder='passthrough')


# In[12]:


x_train_transformed = transformer.fit_transform(x_train)


# In[13]:


x_train_transformed


# In[14]:


x_test_transformed = transformer.fit_transform(x_test)


# In[15]:


x_test_transformed


# In[16]:


y_train


# In[17]:


le = LabelEncoder()


# In[18]:


le.fit(y_train)


# In[19]:


y_train_transformed = le.transform(y_train)


# In[20]:


y_train_transformed


# In[21]:


le.fit(y_test)


# In[22]:


y_test_transformed = le.transform(y_test)


# In[23]:


y_test_transformed


# In[ ]:




