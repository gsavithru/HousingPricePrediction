#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[3]:


import io
get_ipython().run_line_magic('cd', 'C:\\Users\\savit\\Desktop\\Datasets')


# In[4]:


df=pd.read_csv("housing.csv")


# In[6]:


df


# In[7]:


df.info()


# In[8]:


df.dropna()


# In[9]:


df.dropna(inplace=True)


# In[10]:


df.info()


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x=df.drop(['median_house_value'],axis=1)
y=df['median_house_value']


# In[14]:


x


# In[15]:


y


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[25]:


train_df=x_train.join(y_train)


# In[26]:


train_df


# In[27]:


train_df.hist(figsize=(15,8))


# In[29]:


train_df.corr()


# In[30]:


train_df['total_rooms']=np.log(train_df['total_rooms']+1)
train_df['total_bedrooms']=np.log(train_df['total_bedrooms']+1)
train_df['population']=np.log(train_df['population']+1)
train_df['households']=np.log(train_df['households']+1)


# In[31]:


train_df.hist(figsize=(15,8))


# In[32]:


train_df.ocean_proximity.value_counts()


# In[34]:


train_df.join(pd.get_dummies(train_df.ocean_proximity))


# In[35]:


train_df= train_df.join(pd.get_dummies(train_df.ocean_proximity)).drop(['ocean_proximity'],axis=1)


# In[36]:


train_df


# In[37]:


train_df.corr()


# In[38]:


plt.figure(figsize=(15,8))
sns.heatmap(train_df.corr(),annot=True,cmap="YlGnBu")


# In[39]:


train_df['bedroom_ratio']=train_df['total_bedrooms']/train_df['total_rooms']
train_df['household_rooms']=train_df['total_rooms']/train_df['households']


# In[40]:


plt.figure(figsize=(15,8))
sns.heatmap(train_df.corr(),annot=True,cmap="YlGnBu")


# In[41]:


# Simple Linear Regression Model

from sklearn.linear_model import LinearRegression 
x_train, y_train=train_df.drop(['median_house_value'],axis=1),train_df['median_house_value']

reg=LinearRegression()
reg.fit(x_train,y_train)


# In[42]:


test_df=x_test.join(y_test)

test_df['total_rooms']=np.log(test_df['total_rooms']+1)
test_df['total_bedrooms']=np.log(test_df['total_bedrooms']+1)
test_df['population']=np.log(test_df['population']+1)
test_df['households']=np.log(test_df['households']+1)

test_df= test_df.join(pd.get_dummies(test_df.ocean_proximity)).drop(['ocean_proximity'],axis=1)

test_df['bedroom_ratio']=test_df['total_bedrooms']/test_df['total_rooms']
test_df['household_rooms']=test_df['total_rooms']/test_df['households']


# In[43]:


test_df


# In[44]:


train_df

reg.score(x_test,y_test)
# In[46]:


x_test, y_test=test_df.drop(['median_house_value'],axis=1),test_df['median_house_value']


# In[47]:


reg.score(x_test,y_test)


# In[88]:


# RANDOM FOREST MODEL

from sklearn.ensemble import RandomForestRegressor


# In[89]:


forest=RandomForestRegressor()


# In[90]:


forest.fit(x_train,y_train)


# In[91]:


forest.score(x_test,y_test)


# In[ ]:




