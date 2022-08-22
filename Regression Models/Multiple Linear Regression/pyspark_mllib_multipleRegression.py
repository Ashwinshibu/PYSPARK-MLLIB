#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()

#Necessary Python Libraries
import pandas as pd


# In[2]:


data = spark.read.csv(r"C:\Users\ABIN\Desktop\PYSPARK MLLIB\Restaurant_Profit_Data.csv", header=True, inferSchema=True)


# In[3]:


data.show()


# In[4]:


categorical_cols = [item[0] for item in data.dtypes if item[1].startswith('string')]
print(categorical_cols)

numerical_cols = [item[0] for item in data.dtypes if item[1].startswith('int') | item[1].startswith('double')][:-1]
print(numerical_cols)

#Print number of categorical as well as numerical features.
print(str(len(categorical_cols)) + '  categorical features')
print(str(len(numerical_cols)) + '  numerical features')


# In[5]:


from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
stages = []
for categoricalCol in categorical_cols:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    OHencoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "_catVec"])
stages += [stringIndexer, OHencoder]
assemblerInputs = [c + "_catVec" for c in categorical_cols] + numerical_cols
Vectassembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [Vectassembler]


# In[6]:


print(stringIndexer.getOutputCol())


# In[9]:


from pyspark.ml import Pipeline


# In[10]:


cols=data.columns


# In[11]:


cols.show()


# In[12]:


print(cols)


# In[13]:


pipeline=Pipeline(stages=stages)


# In[16]:


pipelineModel=pipeline.fit(data)


# In[18]:


data=pipelineModel.transform(data)


# In[19]:


data.show()


# In[21]:


selected_cols=['features']+cols


# In[22]:


selected_cols.show()


# In[23]:


data=data.select(selected_cols)


# In[24]:


data.show()


# In[25]:


pd.DataFrame(data.take(5), columns=data.columns)


# In[26]:


finalized_data=data.select("features","Profit")


# In[27]:


finalized_data.show()


# In[28]:


train_dataset,test_dataset=finalized_data.randomSplit([0.7,0.3])


# In[29]:


from pyspark.ml.regression import LinearRegression


# In[30]:


linReg=LinearRegression(featuresCol="features",labelCol="Profit")


# In[31]:


model=linReg.fit(train_dataset)


# In[32]:


pred=model.evaluate(test_dataset)


# In[33]:


pred.predictions.show()


# In[34]:


data.show()


# In[35]:


print(model.coefficients)


# In[36]:


print(model.intercept)


# In[37]:


coefficient = model.coefficients
print ("The coefficients of the model are : %a" %coefficient)


# In[39]:


from pyspark.ml.evaluation import RegressionEvaluator


# In[40]:


reg=RegressionEvaluator(labelCol="Profit",predictionCol="prediction")


# In[41]:


rmse=reg.evaluate(pred.predictions,{reg.metricName:"rmse"})


# In[42]:


print(rmse)


# In[50]:


mse=reg.evaluate(pred.predictions,{reg.metricName:"mae"})


# In[51]:


print(mse)


# In[45]:


import numpy as np


# In[48]:


rms=np.sqrt(mse)


# In[49]:


print(rms)


# In[52]:


rmse=reg.evaluate(pred.predictions,{reg.metricName:"r2"})


# In[53]:


print(rmse)


# In[55]:


test_dataset.show()


# In[56]:


unlabeled=test_dataset.select("features")


# In[57]:


newpredictions=model.transform(unlabeled)


# In[58]:


newpredictions.show()


# In[61]:


newDf=test_dataset.join(newpredictions,on="features")


# In[62]:


newDf.show()


# In[81]:


newDf= newDf.withColumn("diff",newDf["Profit"] -newDf["prediction"])


# In[84]:


from pyspark.sql.functions import abs


# In[85]:


newDf=newDf.withColumn("diff",abs("diff"))


# In[86]:


newDf.show()


# In[ ]:




