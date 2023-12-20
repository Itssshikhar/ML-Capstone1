#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests


# In[2]:


url = 'http://localhost:6969/predict'


# In[3]:


patient = {
   "age": 67,
   "sex_": 0,
   "chest_pain_type": 3,
   "resting_blood_pressure": 115,
   "serum_cholestoral": 564,
   "fasting_blood_sugar": 0,
   "resting_electrocardiographic_results": 2,
   "max_heart_rate": 160,
   "exercise_induced_angina": 0,
   "oldpeak": 1
}


# In[4]:


patient


# In[6]:


response = requests.post(url, json=patient).json()
response


# In[ ]:




