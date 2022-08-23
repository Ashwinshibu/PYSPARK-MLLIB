#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()


# In[2]:


# Import necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


# In[4]:


# Import the Height Weight Dataset 
data=pd.read_csv(r"C:\Users\ABIN\Desktop\PYSPARK MLLIB\Height_weight_data.csv")


# In[14]:


data.head()


# In[15]:


#Store the data in the form of dependent and independent variables separately
x=data.iloc[:,0:1].values


# In[26]:


y=data.iloc[:,1].values


# In[27]:


y


# In[28]:


#Split the Dataset into Training and Test Dataset
from sklearn.model_selection import train_test_split


# In[35]:


x_train,x_test=train_test_split(x,test_size=0.3,random_state=0)


# In[37]:


y_train,y_test=train_test_split(y,test_size=0.3,random_state=0)


# In[40]:


print(y_train)


# In[41]:


print(y_test)


# In[42]:


#Fit the Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
LR=LinearRegression()


# In[43]:


LR.fit(x_train,y_train)


# In[46]:


# Visualise the Linear Regression Result
plot.scatter(x_train,y_train,color="Green")
plot.show()


# In[53]:


plot.scatter(x_train,y_train,color="Green")
plot.plot(x_train,LR.predict(x_train),color="Blue")
plot.xlabel("Age")
plot.ylabel("Height")
plot.title("Linear Regression")
plot.show()


# In[63]:


# Add the polynomial term to the equation/model
from sklearn.preprocessing import PolynomialFeatures 
  
polynom = PolynomialFeatures(degree = 2) 
x_polynom = polynom.fit_transform(x_train) 
  
x_polynom


# In[64]:


#Fit the Polynomial Regression Model 
PolyReg = LinearRegression() 
PolyReg.fit(x_polynom, y_train) 


# In[66]:


# Visualise the Polynomial Regression Results 
plot.scatter(x_train, y_train, color = 'green') 
  
plot.plot(x_train, PolyReg.predict(polynom.fit_transform(x_train)), color = 'blue') 
plot.title('Polynomial Regression') 
plot.xlabel('Age') 
plot.ylabel('Height') 
  
plot.show() 


# In[67]:


#Predicted Height from test dataset w.r.t Simple Linear Regression
y_predicted=LR.predict(x_test)


# In[68]:


y_predicted


# In[69]:



#Model Evaluation using R-Square for Simple Linear Regression
from sklearn import metrics
r2=metrics.r2_score(y_test,y_predicted)
print(r2)


# In[70]:


#Model Evaluation using R-Square for polynomial Regression
r2=metrics.r2_score(y_test,PolyReg.predict(polynom.fit_transform(x_test)))
print(r2)


# In[72]:


# Predicting Height based on Age using Linear Regression 
LR.predict([[53]]) 


# In[73]:


# Predicting Height based on Age using Polynomial Regression 
PolyReg.predict(polynom.fit_transform([[53]])) 


# In[ ]:




