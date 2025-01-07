#!/usr/bin/env python
# coding: utf-8

# ## CO2 Dataset from 2010 to 2016

# #**CO2 POLLUTION DATASETS DATA PREPARATION, GRAPHS AND DATA TRAIN, TEST AND SPLIT**
# 
# ---
# 
# 
# 
# #**Part A: UPLOADING DATASETS TO GOOGLE DRIVE AND TO IMPORT LIBRARIES**

# ## Import Libraries

# In[54]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm


# ## Selecting and recording the Datasets. Changing file formats for Google Colab**

# In[55]:


# **Path for CO2 Passenger Car Datasets

path_co2_2016 = 'H:\G__CO2_union'


df_CO2_2016 = pd.read_csv(path_co2_2016,sep=',', encoding='utf-8', on_bad_lines='skip', low_memory = False)


df = df_CO2_2016.copy(deep = True)

df['year'] = df['year'].astype(str)
df['year'] = pd.to_datetime(df['year'])


# In[56]:


df['Category_vehicule'].unique()


# In[57]:


df.isnull().sum().sort_values(ascending = False)*100/len(df)


# In[58]:


df['Fuel_mode'].unique()
df['Fuel_type'].value_counts()


# In[59]:


#df.drop_duplicates(subset=None, keep="first", inplace=True) # Drop duplicates. # Do not use!

# 1) Fill the Nan with the Mode

df['Fuel_mode'] = df['Fuel_mode'].fillna(df['Fuel_mode'].mode()[0]) # fill blank with mode
# Here we filled the NaN with the Mode
df['Mass_in_running_kg'] = df['Mass_in_running_kg'].fillna(df['Mass_in_running_kg'].mode()[0]) # mode, mean, median,min,max
# Here we filled the NaN with the Mode
df['Wheel_base_mm'] = df['Wheel_base_mm'].fillna(df['Wheel_base_mm'].mode()[0]) # mode, mean, median,min,max
df['CO2_emission'] = df['CO2_emission'].fillna(df['CO2_emission'].median()) # mode, mean, median,min,max


#1.5 ) Fill the cero in CO2_emission


mean_value = df['CO2_emission'][df['CO2_emission'] != 0].mean()
df['CO2_emission'] = df['CO2_emission'].replace(0, mean_value)


# 2) Fill the Nan with the Mean

df['Mass_in_running_kg'] = df['Mass_in_running_kg'].fillna(df['Mass_in_running_kg'].mean()) # mode, mean, median,min,max
df['Wheel_base_mm'] = df['Wheel_base_mm'].fillna(df['Wheel_base_mm'].mean()) # mode, mean, median,min,max
df['Axle_width_steering_axle_mm'] = df['Axle_width_steering_axle_mm'].fillna(df['Axle_width_steering_axle_mm'].mean()) # mode, mean, median,min,max

df['Engine_capacity_cm3'] = df['Engine_capacity_cm3'].fillna(df['Engine_capacity_cm3'].mean()) # mode, mean, median,min,max
df['Engine_power_kw'] = df['Engine_power_kw'].fillna(df['Engine_power_kw'].mean()) # mode, mean, median,min,max

# 3) Filter CO2_union with are going to work with only the cars from the category M1

df = df[df['Category_vehicule'] != 'M1G'] # Only category M1
df = df[df['Category_vehicule'] != 'N1G']
df = df[df['Category_vehicule'] != 'N1']
df = df[df['Category_vehicule'] != 'nan']
#df = df[df['Category_vehicule'] != 'M1']

df['Category_vehicule'] = df['Category_vehicule'].fillna(df['Category_vehicule'].mode()[0]) # mode, mean, median,min,max

'''
fuel_type_col = 'Fuel_type'

df = df[df[fuel_type_col] != 'Electric']  # All the electric carburant are filtered
df= df[df[fuel_type_col] != 'Petrol_Electric']
df = df[df[fuel_type_col] != 'Diesel_Electric']
df = df[df[fuel_type_col] != 'Hybrid_Petrol_E']

'''

# 4) All the vehicules below 75 gr/km and beyond 240 gr/km will be filtered from CO2_union

'''

df = df[(df['CO2_emission'] >=80) & (df['CO2_emission']<=220)]

# 5) Engine Capacity Filtering

'Engine_capacity_cm3'

df = df[(df['Engine_capacity_cm3'] >=1700) & (df['Engine_capacity_cm3']<=2100)]

# 6) Mass in running KG Filtering

df = df[(df['Mass_in_running_kg'] >=900) & (df['Mass_in_running_kg']<=2000)]

'''

#7) Drop Manufacturing Pooling

#Manufacturer_pooling           33.002838
df = df.drop('Manufacturer_pooling', axis=1)

#8) Fill the Nan with the Mode

df['Make'] = df['Make'].fillna(df['Make'].mode()[0])
df['Member_state'] = df['Member_state'].fillna(df['Member_state'].mode()[0])
df['Manufacturer_name_om'] = df['Manufacturer_name_om'].fillna(df['Manufacturer_name_om'].mode()[0])


#9) Fill the Nan with the Mean

df['co2_regnum'] = df['co2_regnum'].fillna(df['co2_regnum'].mean())
df['Total_new_registration'] = df['Total_new_registration'].fillna(df['Total_new_registration'].mean())



# In[60]:


df.isnull().sum().sort_values(ascending = False)*100/len(df)
#df.info()


# In[61]:


df


# In[ ]:


fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10,6))

sns.histplot(df['CO2_emission'], bins = 20, kde = True, ax =axes[0]);
axes[0].set_title('CO2 emmission per Auto');
axes[0].set_xlabel('CO2 gr/Km ');
axes[0].set_ylabel('Distribution');


sns.boxplot(df['CO2_emission'], ax= axes[1]);
axes[1].set_title('CO2 emmission per Auto');
axes[1].set_xlabel('Frecuency ');
axes[1].set_ylabel('Distribution');


plt.tight_layout();
plt.show();


# In[ ]:


fuel_type_col = 'Fuel_type'


columns = ['Member_state','Manufacturer_name_EU','Manufacturer_name_om',
                   'Manufacturer_name_ms','Type_approval_number','Type','Variant','Version','Make','Commercial_name',
                   'Category_vehicule','Total_new_registration','CO2_emission','Mass_in_running_kg','Wheel_base_mm',
                   'Axle_width_steering_axle_mm','Axle_width_other_axle_mm','Fuel_type','Fuel_mode','Engine_capacity_cm3',
                   'Engine_power_kw']



# Ensure the columns exist in the DataFrame
columns = [col for col in columns if col in df.columns]

# Select relevant variables and drop rows with missing CO2 values
filtered_df = df[columns].dropna(subset=['CO2_emission'])

# Ensure no duplicate columns exist in filtered_df
assert filtered_df.columns.duplicated().sum() == 0, "There are still duplicate columns in filtered_df"

filtered_df = filtered_df.reset_index() # Julio Mella code

# Step 1: Identify the car that emits the most CO2 emissions
max_co2 = filtered_df['CO2_emission'].max()
max_co2_car = filtered_df[filtered_df['CO2_emission'] == max_co2]

print("Car with the highest CO2 emissions:")
print(max_co2_car[['Make', 'CO2_emission']])

# Step 2: Analyze correlations between CO2 emissions and other numerical variables
numerical_columns = ['CO2_emission','Mass_in_running_kg','Wheel_base_mm',
                   'Axle_width_steering_axle_mm','Engine_capacity_cm3',
                   'Engine_power_kw']

# Ensure the numerical columns exist in the DataFrame
numerical_columns = [col for col in numerical_columns if col in filtered_df.columns]

# Step 3: Create a scatter plot matrix to visualize relationships between CO2 and other variables
sns.pairplot(filtered_df[numerical_columns])
plt.suptitle('Scatter Plot Matrix of CO2 and Related Variables', y=1.02)
plt.show()

# Create a heatmap to visualize the correlation matrix
correlation_matrix = filtered_df[numerical_columns].corr()

plt.figure(figsize=(14, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix with CO2 Emissions')
plt.show()

# Print the correlation matrix to identify the variables that correlate the most with CO2 emissions
print("Correlation matrix with CO2 emissions:")
print(correlation_matrix['CO2_emission'].sort_values(ascending=False))

# Reset the index of filtered_df to ensure unique index values
filtered_df = filtered_df.reset_index(drop=True)

# Step 3: Create a scatter plot matrix to visualize relationships between CO2 and other variables
sns.pairplot(filtered_df[numerical_columns])
plt.suptitle('Scatter Plot Matrix of CO2 and Related Variables', y=1.02)
plt.show()


# In[ ]:


scat = plt.scatter(df['CO2_emission'], df['Mass_in_running_kg'],
                   c = df['Engine_capacity_cm3'], cmap=plt.cm.rainbow,
                   s = df['Wheel_base_mm'], alpha = 0.4);

# Color scheme

plt.colorbar(scat, label = 'Mass in running Kg');
plt.rcParams['figure.dpi']= 120

plt.xlabel('CO2 emission');
plt.ylabel('Engine capacity cm3');

plt.legend(['Size : Wheel Base','Color : Mass in running KG'], loc = 'upper left')

plt.title('CO2_emission / Engine_capacity_cm3');
plt.grid(True);


# In[62]:


df['CO2_class'] = " "


# In[63]:


df.loc[(df['CO2_emission']<=100),'CO2_class']='A'
df.loc[(df['CO2_emission']>100) & (df['CO2_emission']<=120) ,'CO2_class']='B'
df.loc[(df['CO2_emission']>120) & (df['CO2_emission']<=140) ,'CO2_class']='C'
df.loc[(df['CO2_emission']>140) & (df['CO2_emission']<=160) ,'CO2_class']='D'
df.loc[(df['CO2_emission']>160) & (df['CO2_emission']<=200) ,'CO2_class']='E'
df.loc[(df['CO2_emission']>200) & (df['CO2_emission']<=250) ,'CO2_class']='F'
df.loc[(df['CO2_emission']>250),'CO2_class']='G'


# In[64]:


df['CO2_class'].value_counts()


# ## Prepare Dataframe For Time Series

# In[65]:


df.info()


# In[66]:


df_time_series = df[['year','CO2_emission']]



# In[67]:


df_time_series = df_time_series.set_index('year')
#df_time_series = df_time_series.asfreq('MS')


# In[68]:


df_time_series.index


# In[69]:


plt.plot(df_time_series)
plt.show()


# In[70]:


from statsmodels.tsa.seasonal import seasonal_decompose

res = seasonal_decompose(df_time_series, model="additive", period=12)
res.plot()
plt.show()


# In[71]:


res = seasonal_decompose(df_time_series, model = 'multiplicative', period = 12)
res.plot()
plt.show()


# 
# ## Filter df  Dataframe by class 

# In[72]:


df_A = df[df['CO2_class'] == "A"] 
df_B = df[df['CO2_class'] == "B"] 
df_C = df[df['CO2_class'] == "C"] 
df_D = df[df['CO2_class'] == "D"] 
df_E = df[df['CO2_class'] == "E"] 
df_F = df[df['CO2_class'] == "F"] 
df_G = df[df['CO2_class'] == "G"] 


# In[73]:


df_time_series_A = df_A[['year','CO2_emission']]
df_time_series_B = df_B[['year','CO2_emission']]
df_time_series_C = df_C[['year','CO2_emission']]
df_time_series_D = df_D[['year','CO2_emission']]
df_time_series_E = df_E[['year','CO2_emission']]
df_time_series_F = df_F[['year','CO2_emission']]
df_time_series_G = df_G[['year','CO2_emission']]



# In[74]:


df_time_series_A = df_time_series_A.set_index('year')
df_time_series_B = df_time_series_B.set_index('year')
df_time_series_C = df_time_series_C.set_index('year')
df_time_series_D = df_time_series_D.set_index('year')
df_time_series_E = df_time_series_E.set_index('year')
df_time_series_F = df_time_series_F.set_index('year')
df_time_series_G = df_time_series_G.set_index('year')



# In[86]:


res = seasonal_decompose(df_time_series_A, model = 'multiplicative', period = 12)
res2 = seasonal_decompose(df_time_series_B, model = 'multiplicative', period = 12)
res3 = seasonal_decompose(df_time_series_C, model = 'multiplicative', period = 12)
res4 = seasonal_decompose(df_time_series_D, model = 'multiplicative', period = 12)
res5 = seasonal_decompose(df_time_series_E, model = 'multiplicative', period = 12)
res6 = seasonal_decompose(df_time_series_F, model = 'multiplicative', period = 12)
res7 = seasonal_decompose(df_time_series_G, model = 'multiplicative', period = 12)



#res3.plot()
#plt.show()


# In[84]:


fig, axes = plt.subplots(4, 3, figsize=(15, 10))

# Plot first time series decomposition
axes[0, 0].plot(res.observed)
axes[0, 0].set_title('Observed (A)')
axes[1, 0].plot(res.trend)
axes[1, 0].set_title('Trend (A)')
axes[2, 0].plot(res.seasonal)
axes[2, 0].set_title('Seasonal (A)')
axes[3, 0].plot(res.resid)
axes[3, 0].set_title('Residual (A)')

# Plot second time series decomposition
axes[0, 1].plot(res2.observed)
axes[0, 1].set_title('Observed (B)')
axes[1, 1].plot(res2.trend)
axes[1, 1].set_title('Trend (B)')
axes[2, 1].plot(res2.seasonal)
axes[2, 1].set_title('Seasonal (B)')
axes[3, 1].plot(res2.resid)
axes[3, 1].set_title('Residual (B)')


# Plot third time series decomposition
axes[0, 2].plot(res3.observed)
axes[0, 2].set_title('Observed (C)')
axes[1, 2].plot(res3.trend)
axes[1, 2].set_title('Trend (C)')
axes[2, 2].plot(res3.seasonal)
axes[2, 2].set_title('Seasonal (C)')
axes[3, 2].plot(res3.resid)
axes[3, 2].set_title('Residual (C)')


plt.tight_layout()
plt.show()


# In[88]:


fig, axes = plt.subplots(4, 4, figsize=(15, 10))

# Plot fourth time series decomposition
axes[0, 0].plot(res4.observed)
axes[0, 0].set_title('Observed (D)')
axes[1, 0].plot(res4.trend)
axes[1, 0].set_title('Trend (D)')
axes[2, 0].plot(res4.seasonal)
axes[2, 0].set_title('Seasonal (D)')
axes[3, 0].plot(res4.resid)
axes[3, 0].set_title('Residual (D)')

# Plot fifth time series decomposition
axes[0, 1].plot(res5.observed)
axes[0, 1].set_title('Observed (E)')
axes[1, 1].plot(res5.trend)
axes[1, 1].set_title('Trend (E)')
axes[2, 1].plot(res5.seasonal)
axes[2, 1].set_title('Seasonal (E)')
axes[3, 1].plot(res5.resid)
axes[3, 1].set_title('Residual (E)')


# Plot sixth time series decomposition
axes[0, 2].plot(res6.observed)
axes[0, 2].set_title('Observed (F)')
axes[1, 2].plot(res6.trend)
axes[1, 2].set_title('Trend (F)')
axes[2, 2].plot(res6.seasonal)
axes[2, 2].set_title('Seasonal (F)')
axes[3, 2].plot(res6.resid)
axes[3, 2].set_title('Residual (F)')


# Plot sixth time series decomposition
axes[0, 3].plot(res7.observed)
axes[0, 3].set_title('Observed (G)')
axes[1, 3].plot(res7.trend)
axes[1, 3].set_title('Trend (G)')
axes[2, 3].plot(res7.seasonal)
axes[2, 3].set_title('Seasonal (G)')
axes[3, 3].plot(res7.resid)
axes[3, 3].set_title('Residual (G)')

plt.tight_layout()
plt.show()


# In[ ]:




