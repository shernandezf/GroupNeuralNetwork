#!/usr/bin/env python
# coding: utf-8

# In[108]:


import numpy as np


# In[109]:


import pandas as pd


# In[110]:


import tensorflow as tf


# In[111]:


import matplotlib.pyplot as plt


# In[113]:


train_dataset = pd.read_csv('D:\\NNetwork group task\\train.csv', parse_dates=['Date'], low_memory=False)
train_dataset


# In[115]:


store_dataset = pd.read_csv('D:\\NNetwork group task\\store.csv')
store_dataset


# In[116]:


merged_dataset = pd.merge(train_dataset, store_dataset, on = "Store", how = "left" )
merged_dataset


# In[117]:


merged_dataset.isnull().sum()


# In[118]:


CompetitionDistance = 'CompetitionDistance'
Promo2SinceWeek ='Promo2SinceWeek' 
CompetitionOpenSinceMonth = 'CompetitionOpenSinceMonth'
CompetitionOpenSinceYear = 'CompetitionOpenSinceYear'
Promo2SinceYear = 'Promo2SinceYear'
PromoInterval = 'PromoInterval'

if pd.api.types.is_numeric_dtype(merged_dataset[PromoInterval]):
    # Fill NaN values with the mean for numeric columns
    merged_dataset[PromoInterval].fillna(merged_dataset[PromoInterval].mean(), inplace=True)
else:
    # For non-numeric columns, fill NaN values with the most frequent value (mode)
    merged_dataset[PromoInterval].fillna(0, inplace=True)


# In[119]:


merged_dataset[CompetitionDistance].fillna(merged_dataset[CompetitionDistance].mean(), inplace=True)
merged_dataset[Promo2SinceWeek].fillna(merged_dataset[Promo2SinceWeek].mean(), inplace=True)
merged_dataset[CompetitionOpenSinceMonth].fillna(merged_dataset[CompetitionOpenSinceMonth].mean(), inplace=True)
merged_dataset[CompetitionOpenSinceYear].fillna(merged_dataset[CompetitionOpenSinceYear].mean(), inplace=True)
merged_dataset[Promo2SinceYear].fillna(merged_dataset[Promo2SinceYear].mean(), inplace=True)
#merged_dataset[PromoInterval].fillna(merged_dataset[PromoInterval].mean(), inplace=True)
#merged_dataset["PromoInterval"] = merged_dataset["PromoInterval"].map(promo_interval_mapping)


# In[120]:


merged_dataset


# In[121]:


merged_dataset.isnull().sum()


# In[122]:


from sklearn import preprocessing
lb=preprocessing.LabelEncoder()

merged_dataset['StoreType']=lb.fit_transform(merged_dataset['StoreType'])
merged_dataset['Assortment']=lb.fit_transform(merged_dataset['Assortment'])
merged_dataset['StateHoliday']=lb.fit_transform(merged_dataset['StateHoliday'])

merged_dataset['PromoInterval'] = merged_dataset['PromoInterval'].astype(str)
merged_dataset['PromoInterval']=lb.fit_transform(merged_dataset['PromoInterval'])
print(merged_dataset.columns)

#print(merged_dataset['PromoInterval']).unique()


merged_dataset.dtypes


# In[123]:


merged_dataset.isnull().sum()


# In[124]:


merged_dataset


# In[125]:


import seaborn as sns


# In[127]:


numeric_data = merged_dataset.select_dtypes(include=['int64', 'float64'])

correlation_matrix = numeric_data.corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap for Rossman Sales Data')
plt.show()


# In[133]:


from sklearn.preprocessing import MinMaxScaler


# In[134]:


numeric_data = merged_dataset.select_dtypes(include=['int64', 'float64'])


# In[139]:


scaling_parameter = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear', 'DayOfWeek','Customers', 'Sales']


scaler = MinMaxScaler()

# Apply Min-Max scaling to selected columns
merged_dataset[scaling_parameter] = scaler.fit_transform(merged_dataset[scaling_parameter])

merged_dataset[scaling_parameter]


# In[ ]:




