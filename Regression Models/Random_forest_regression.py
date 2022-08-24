#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


data=pd.read_csv(r"C:\Users\ABIN\Desktop\PYSPARK MLLIB\Height_age_data.csv")


# In[4]:


data


# In[15]:


x=data.iloc[:,0:1].values


# In[23]:


x.shape


# In[9]:


y=data.iloc[:,1].values


# In[24]:


y.shape


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[18]:


from sklearn.ensemble import RandomForestRegressor


# In[37]:


RF=RandomForestRegressor(n_estimators=500,random_state=0)


# In[38]:


RF.fit(x_train,y_train)


# In[39]:


y_predict=RF.predict(x_test)


# In[40]:


y_predict


# In[41]:


from sklearn import metrics
r2=metrics.r2_score(y_test,y_predict)


# In[42]:


r2


# In[43]:


plt.scatter(x_train,y_train,color="green")
x_val=np.arange(min(x_train),max(x_train),0.01)
x_val.shape


# In[44]:


x_val=x_val.reshape(len(x_val),1)


# In[45]:


x_val


# In[46]:


plt.plot(x_val,RF.predict(x_val),color="blue")
plt.scatter(x_train,y_train,color="green")
plt.title("Random Forest")
plt.xlabel("Age")
plt.ylabel("Height")
plt.figure(figsize=(1,1))
plt.show()


# In[47]:


height_pred=RF.predict([[41]])


# In[48]:


height_pred


# In[ ]:




