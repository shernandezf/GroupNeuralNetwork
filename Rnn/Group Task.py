#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install keras


# In[1]:


pip install tensorflow


# In[2]:


from keras.models import Sequential


# In[4]:


from keras.layers import Dense


# In[5]:


from keras.layers import LSTM


# In[9]:


from keras.models import Sequential


# In[10]:


reg = Sequential()


# In[15]:


import numpy as np


# In[16]:


import pandas as pd


# In[17]:


import matplotlib.pyplot as plt


# In[21]:


train_set = pd.read_csv('D:\\Group Task\\train.csv', dtype={'column_name':'str'}, low_memory = False)


# In[22]:


train_set


# In[23]:


train_set = train_set.iloc[:,3:4].values


# In[24]:


from sklearn.preprocessing import MinMaxScaler


# In[25]:


sc = MinMaxScaler()


# In[26]:


train_set = sc.fit_transform(train_set)


# In[28]:


len(train_set)


# In[29]:


x_train = train_set[0:1017208]


# In[32]:


y_train = train_set[1:1017209]


# In[33]:


x_train = np.reshape(x_train,(1017208,1,1))


# In[34]:


x_train


# In[59]:


reg = Sequential()


# In[63]:


x_train_reshaped = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))


# In[66]:


reg.add(LSTM(units=100,input_shape=(2,3),return_sequences=True))


# In[ ]:




