#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[3]:


data = pd.read_csv('C:\\Users\\MamunaSafdar\\Downloads\\WA_Fn-UseC_-HR-Employee-Attrition.CSV')


# In[4]:


print("Shape of the dataset:", data.shape)
print("\nColumns in the dataset:", data.columns)


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


print("Missing values:\n", data.isnull().sum())


# In[8]:


data.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)


# In[9]:


data_encoded = pd.get_dummies(data, drop_first=True)


# In[10]:


X = data_encoded.drop('Attrition_Yes', axis=1)  
y = data_encoded['Attrition_Yes']


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[13]:


clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)


# In[14]:


y_pred = clf.predict(X_test_scaled)


# In[15]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[16]:


X = data.drop(columns=['Attrition'])
y = data['Attrition']


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


# In[20]:


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[21]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# In[22]:


model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42))])


# In[23]:


model.fit(X_train, y_train)


# In[24]:


y_pred = model.predict(X_test)


# In[25]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[26]:


print("Classification Report:\n", classification_report(y_test, y_pred))

