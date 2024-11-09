#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install matplotlib')




# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10,6)
import warnings
warnings.filterwarnings('ignore')


# In[5]:


df = pd.read_csv('Advertising.csv')
df


# In[6]:


df.info()


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.sample(5)


# In[10]:


df.describe().T


# In[11]:


df.corr()


# In[12]:


sns.heatmap(df.corr(), annot=True);


# In[13]:


df['total_spend'] = df.TV + df.radio + df.newspaper
df


# In[15]:


df = df.iloc[:, [0,1,2,4,3]]
df


# In[16]:


sns.pairplot(df)


# In[17]:


pip install scipy


# In[18]:


import scipy.stats as stats


# In[19]:


for i in df.drop(columns = 'sales'):
    print(f'corr between sales and{i:<12}: {df.sales.corr(df[i])}')


# In[20]:


sns.heatmap(df.corr(), annot=True);


# In[21]:


df = df[['total_spend', 'sales']]
df.head()


# In[22]:


sns.scatterplot(x='total_spend', y ='sales', data=df);


# In[23]:


corr = df.sales.corr(df.total_spend)
corr


# In[24]:


df['total_spend'].corr(df['sales'])


# In[25]:


R2_score = corr**2
R2_score


# In[26]:


sns.regplot(x='total_spend', y='sales', data = df, ci=None) ;


# In[27]:


X = df['total_spend']
y=df['sales']


# In[28]:


np.polyfit(X, y, deg=1)


# In[29]:


slope, intercept = np.polyfit(X,y, deg=1)


# In[30]:


print('slope       :',  slope)
print('intercept   :',  intercept)


# In[31]:


b1, b0 = np.polyfit(X,y, deg = 1)


# In[32]:


print('b1 :', b1)
print('b0 :', b0)


# In[34]:


y_pred = b1*X + b0


# In[35]:


y_pred


# In[37]:


values = {'actual': y, 'predicted': y_pred, 'residual':y-y_pred, 'LSE': (y-y_pred)**2}
df_2 = pd.DataFrame(values)
df_2


# In[38]:


df_2.residual.sum().round()


# In[39]:


df_2.LSE


# In[40]:


df_2.LSE.sum()


# In[41]:


potential_spend = np.linspace(0,500,100)
potential_spend


# In[44]:


predicted_sales_lin = b1*potential_spend + b0
predicted_sales_lin


# In[45]:


plt.plot(potential_spend, predicted_sales_lin)


# In[47]:


plt.plot(potential_spend, predicted_sales_lin)
sns.scatterplot(x='total_spend', y='sales', data=df);


# In[48]:


a = np.polyfit(X, y, deg=3)
a


# In[49]:


a1 = np.polyfit(X,y, deg=3)[0]
a2 = np.polyfit(X,y, deg=3)[1]
a3 = np.polyfit(X,y, deg=3)[2]
a0 = np.polyfit(X,y, deg=3)[3]


# In[51]:


predicted_sales_poly = a1*potential_spend**3 + a2*potential_spend**2+a3*potential_spend + a0
predicted_sales_poly


# In[53]:


plt.plot(potential_spend, predicted_sales_poly, color='pink');


# In[54]:


plt.plot(potential_spend, predicted_sales_poly, color='pink')
sns.scatterplot(x='total_spend', y= 'sales', data=df);


# In[55]:


plt.plot(potential_spend, predicted_sales_poly, color='pink')
plt.plot(potential_spend, predicted_sales_lin, color='yellow')

sns.scatterplot(x='total_spend', y= 'sales', data=df);


# In[57]:


spend = 400
sales_pred_lin = b1*spend + b0
sales_pred_poly = a1*spend**3 + a2*spend**2 +a3*spend**a0

print(sales_pred_lin, sales_pred_poly)


# In[58]:


z = np.polyfit(X, y, deg=10)
z


# In[59]:


z1 = np.polyfit(X,y, deg = 10) [0]
z2 = np.polyfit(X,y, deg = 10) [1]
z3 = np.polyfit(X,y, deg = 10) [2]
z4 = np.polyfit(X,y, deg = 10) [3]
z5 = np.polyfit(X,y, deg = 10) [4]
z6 = np.polyfit(X,y, deg = 10) [5]
z7 = np.polyfit(X,y, deg = 10) [6]
z8 = np.polyfit(X,y, deg = 10) [7]
z9 = np.polyfit(X,y, deg = 10) [8]
z10 = np.polyfit(X,y, deg = 10) [9]
z0 = np.polyfit(X,y, deg = 10) [0]


# In[60]:


pred_lin = b1 * X +b0
pred_poly = z1 * X **10 +z2 * X **9 +z3 * X **8 +z4 * X **7 +z5 * X **6 +z6 * X **5 +z7 * X **4 +z8 * X **3 +z9 * X **2 +z10 * X +z0
print(pred_lin, pred_poly)


# In[61]:


values = {'actual':y, 'predicted': pred_poly, 'resudial': y-pred_poly, 'LSE': (y-pred_poly)**2}
df_pol = pd.DataFrame(values)
df_pol


# In[63]:


df_pol.LSE.sum()


# In[64]:


values_lin = {'actual':y, 'predicted': pred_lin, 'resudial': y-pred_lin, 'LSE': (y-pred_lin)**2}
df_lin = pd.DataFrame(values_lin)
df_lin


# In[65]:


df_lin.LSE.sum()

