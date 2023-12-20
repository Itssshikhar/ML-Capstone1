#!/usr/bin/env python
# coding: utf-8

# ## Heart Disease Prediction

# - Download the data from this
# 
#   https://www.kaggle.com/datasets/utkarshx27/heart-disease-diagnosis-dataset?rvi=1
# 
# - Unzip the file and load the dataset

# ### Loading the Data

# In[4]:


import numpy as np
import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt


# In[5]:


df = pd.read_csv("dataset_heart.csv")
df


# In[6]:


df.head()


# ### Exploratory Data Analysis(EDA)

# In[7]:


columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'resting electrocardiographic results', 'max heart rate', 'exercise induced angina', 'oldpeak', 'ST segment', 'major vessels', 'thal', 'heart disease']


# In[8]:


df.columns = map(str.lower, df.columns)

df.columns = df.columns.str.replace(' ', '_')


# In[9]:


df


# In[10]:


print(df.dtypes)


# In[11]:


numeric_features = list(df.select_dtypes(include=[np.number]).columns)
print(numeric_features)


# In[12]:


target = 'heart_disease'
numeric_features.remove(target)

print(numeric_features)


# In[13]:


df[numeric_features] = df[numeric_features].astype(int)


# In[14]:


print(df[target].value_counts(normalize=True).round(2))

df[target].value_counts().plot(kind='bar', rot=0)
plt.xlabel('Heart Disease')
plt.ylabel('Counts')

for i, count in enumerate(df[target].value_counts()):
    plt.text(i, count-50, count,  ha='center', va='top', fontweight='bold')


# In[15]:


df[target].value_counts().plot(kind='pie', labels=['No heart disease', 'Heart disease'], autopct='%1.1f%%', startangle=90)
plt.ylabel('')
plt.show()


# ### Data Processing and Feature Analysis

# In[16]:


from sklearn.metrics import mutual_info_score


# In[17]:


df.head()


# In[18]:


print(numeric_features)


# In[19]:


feature_importance = []

# Create a dataframe for the analysis
results = pd.DataFrame(columns=['Feature', 'Value', 'Percentage'])

for feature in numeric_features:    
    grouped = df.groupby(feature)[target].mean().reset_index()
    grouped.columns = ['Value', 'Percentage']
    grouped['Feature'] = feature
    results = pd.concat([results, grouped], axis=0)

# Sort the results by percentage in descending order and get the top 10
results = results.sort_values(by='Percentage', ascending=False).head(50)

# get the overall heart diease occurrence rate
overall_rate = df[target].mean()
print('Overall Rate',overall_rate)

# calculate the difference between the feature value percentage and the overall rate
results['Difference'] = results['Percentage'] - overall_rate

# calculate the ratio of the difference to the overall rate
results['Ratio'] = results['Difference'] / overall_rate

# calculate the risk of heart disease occurrence for each feature value
results['Risk'] = results['Percentage'] / overall_rate

# sort the results by ratio in descending order
results = results.sort_values(by='Risk', ascending=False)

print(results)
plt.figure(figsize=(25, 10))
sns.barplot(orient='h', data=results, x='Percentage', y='Value', hue='Feature')
plt.xlabel('Percentage of Heart Disease Occurrences')
plt.ylabel('Feature Value')
plt.title('Top 15 Ranking of Feature Values by Heart Disease Occurrence')
plt.show()


# ### Mutual Information Score

# In[20]:


x = df[numeric_features]
y = df[target]

def mutual_heart_info(series):
    return mutual_info_score(series, y) 

mi_score = x.apply(mutual_heart_info)
mi_ranking = pd.Series(mi_score, index=x.columns).sort_values(ascending=False)

print(mi_ranking)

plt.figure(figsize=(12,6))
sns.barplot(x=mi_ranking.values, y=mi_ranking.index)
plt.xlabel('Importance of Mutual Score')
plt.ylabel('Features')
plt.title('Mutual Info Score on various features')


# ### Model Training


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

target = 'heart_disease'

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.heart_disease.values
y_test = df_test.heart_disease.values
y_val = df_val.heart_disease.values

del df_train['heart_disease']
del df_val['heart_disease']
del df_test['heart_disease']

df_train

# ### Using Different Models

# Using Linear Regression
train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

model = LinearRegression()
model.fit(X_train, y_train)
print(f"Score of the model: {model.score(X_val, y_val)}")

preds = model.predict(X_val)
rmse = mean_squared_error(y_test, preds, squared=False)

print(f"RMSE of the base model: {rmse:.3f}")
print(f"Accuracy Score: {accuracy_score(y_val, y_test)}")


# Using Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X_train, y_train)
print(f"Score of the model: {regressor.score(X_val, y_val)}")

preds = regressor.predict(X_val)
rmse = mean_squared_error(y_test, preds, squared=False)

print(f"RMSE of the base model: {rmse:.3f}")


# Using Random Forest Regressor
regr = RandomForestRegressor(max_depth=2, random_state=0)

regr.fit(X_train, y_train)
print(f"Score of the model: {regr.score(X_val, y_val)}")

preds = regr.predict(X_val)
rmse = mean_squared_error(y_test, preds, squared=False)

print(f"RMSE of the base model: {rmse:.3f}")

# Using XGboost
dtrain_reg = xgb.DMatrix(X_train, y_train)
dtest_reg = xgb.DMatrix(df_test, y_test)

params = {"objective": "reg:squarederror", "tree_method": "hist", "device": "cuda"}
n=100
model = xgb.train(params=params, dtrain=dtrain_reg, num_boost_round=n)

preds = model.predict(dtest_reg)
rmse = mean_squared_error(y_test, preds, squared=False)

print(f"RMSE of the base model: {rmse:.3f}")


# ### Saving and Loading the Model

import pickle

output_file = 'model.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')




