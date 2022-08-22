#!/usr/bin/env python
# coding: utf-8

# Start Spark Session
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()

#Necessary Python Libraries
import pandas as pd


#Load file


data = spark.read.csv(r"C:\Users\ABIN\Desktop\PYSPARK MLLIB\Restaurant_Profit_Data.csv", header=True, inferSchema=True)


#Display few rows


data.show()


#Create features storing categorical & numerical variables, omitting the last column

categorical_cols = [item[0] for item in data.dtypes if item[1].startswith('string')]
print(categorical_cols)

numerical_cols = [item[0] for item in data.dtypes if item[1].startswith('int') | item[1].startswith('double')][:-1]
print(numerical_cols)

#Print number of categorical as well as numerical features.
print(str(len(categorical_cols)) + '  categorical features')
print(str(len(numerical_cols)) + '  numerical features')


# First using StringIndexer to convert string/text values into numerical values followed by OneHotEncoderEstimator 
# Spark MLLibto convert each Stringindexed or transformed values into One Hot Encoded values.
# VectorAssembler is being used to assemble all the features into one vector from multiple columns that contain type double 
# Also appending every step of the process in a stages array


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


# Using a Spark MLLib pipeline to apply all the stages of transformation


from pyspark.ml import Pipeline

cols=data.columns

cols.show()

print(cols)

pipeline=Pipeline(stages=stages)

pipelineModel=pipeline.fit(data)

data=pipelineModel.transform(data)


# Display data

data.show()

selected_cols=['features']+cols

selected_cols.show()

data=data.select(selected_cols)

data.show()

pd.DataFrame(data.take(5), columns=data.columns)


#Select only Features and Label from previous dataset as we need these two entities for building machine learning model


finalized_data=data.select("features","Profit")


# In[27]:


finalized_data.show()


#Split the data into training and test model with 70% obs. going in training and 30% in testing


train_dataset,test_dataset=finalized_data.randomSplit([0.7,0.3])


#Import Linear Regression class called LinearRegression


from pyspark.ml.regression import LinearRegression


#Create the Multiple Linear Regression object named linReg  having feature column as features and Label column as Profit


linReg=LinearRegression(featuresCol="features",labelCol="Profit")



#Train the model on the training using fit() method.

model=linReg.fit(train_dataset)


#Predict the Profit on Test Dataset using the evulate method


pred=model.evaluate(test_dataset)

#Show the predicted Grade values along side actual Grade values

pred.predictions.show()

data.show()


#Find out coefficient value


print(model.coefficients)


#Find out intercept Value


print(model.intercept)


coefficient = model.coefficients
print ("The coefficients of the model are : %a" %coefficient)


#Evaluate the model using metric like Mean Absolute Error(MAE), Root Mean Square Error(RMSE) and R-Square

from pyspark.ml.evaluation import RegressionEvaluator


reg=RegressionEvaluator(labelCol="Profit",predictionCol="prediction")


# root mean square error value


rmse=reg.evaluate(pred.predictions,{reg.metricName:"rmse"})

print(rmse)


# mean square error value

mse=reg.evaluate(pred.predictions,{reg.metricName:"mae"})

print(mse)


# import numpy


import numpy as np


# root mean square value with numpy


rms=np.sqrt(mse)


print(rms)


#  r2 - coefficient of determination


r2=reg.evaluate(pred.predictions,{reg.metricName:"r2"})

print(r2)


#Display test_dataset


test_dataset.show()


##Create Unlabeled dataset  to contain only feature column


unlabeled=test_dataset.select("features")


# Predict the model output for fresh & unseen test data using transform() method


newpredictions=model.transform(unlabeled)


#Display new predictions


newpredictions.show()


# compare test_dataset and newpredictions


newDf=test_dataset.join(newpredictions,on="features")


#Display newDf


newDf.show()


# add a column diff which is difference between actual profit and prediction


newDf= newDf.withColumn("diff",newDf["Profit"] -newDf["prediction"])


# import absolute function


from pyspark.sql.functions import abs


# make values of diff column absolute values


newDf=newDf.withColumn("diff",abs("diff"))


# Display


newDf.show()







