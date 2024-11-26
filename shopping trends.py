#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #vectors, matrices, linear equation, vector spaces, linear transformations
import pandas as pd #data processing


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns #making pretier and easier to read graphs compared to matplotlib
sns.set() #tto apply a default style to the graphs

import warnings
warnings.filterwarnings('ignore') # to stop warnings


# In[5]:


df=pd.read_csv('shopping_trends.csv')
df.head()


# In[8]:


df.shape #rows and columns


# In[10]:


df.info()


# In[15]:


df.isna().sum()


# In[16]:


df.duplicated().sum()


# In[17]:


df.describe()


# In[19]:


df.describe().T


# In[22]:


num_cols = df.select_dtypes(exclude=['O']).columns.to_list() #select columns in the dataframe that are not type 'O' Object(text data) by excluding "O" you only get columns like integers, floating point number
print(f'Numeric Columns ({len(num_cols)}) : \n {num_cols} ') #


# In[24]:


cat_cols = df.select_dtypes(include=['O']).columns.to_list()
print(f'Object Columns ({len(cat_cols)}) : \n {cat_cols}')


# In[25]:


for i, col in enumerate(df.columns):
    print(f'{i+1}. {col} ({df[col].nunique()}) : \n {df[col].unique()} ')


# In[29]:


plt.figure(figsize = (15, 10))
for i, col in enumerate(num_cols):
    plt.subplot(2, 2, i+1)
    sns.histplot(data=df, x=col, hue = 'Gender' , kde=True, palette = 'Set2')
    plt.xlabel(col)
    plt.title(col)
plt.tight_layout()
plt.show


# In[30]:


plt.figure(figsize = (15, 10))
for i, col in enumerate(num_cols):
    plt.subplot(2, 2, i+1)
    sns.histplot(data = df, x = col, hue = 'Gender', kde = True, palette = 'Set2')
    plt.xlabel(col)
    plt.title(col)
plt.tight_layout()
plt.show()


# In[31]:


plt.figure(figsize= (8,6))
sns.heatmap(df[num_cols].corr(), cmap='Blues', annot=True, fmt='.2f')
plt.show()


# In[32]:


df.info()


# In[33]:


df.head()


# In[34]:


purchase_item = df.groupby(['Item Purchased'])[['Purchase Amount (USD)']].sum()
print(purchase_item)
purchase_item.plot(kind='bar')


# In[35]:


category = df.groupby('Category')[['Purchase Amount (USD)']].sum()
print(category)
category.plot(kind='bar')


# In[36]:


gender = df.groupby('Gender')[['Purchase Amount (USD)']].sum()
print(gender)
gender.plot(kind='bar')


# In[37]:


color = df.groupby('Color')[['Purchase Amount (USD)']].sum()
print(color)
color.plot(kind='bar')


# In[38]:


color = df.groupby('Color')[['Customer ID']].count()
print(color)
color.plot(kind='bar')

