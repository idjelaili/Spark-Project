
# coding: utf-8

# # Linear Regression Example
# 

# In[3]:


from pyspark.sql import SparkSession


# In[4]:


spark = SparkSession.builder.appName('lr_example').getOrCreate()


# In[5]:


from pyspark.ml.regression import LinearRegression


# In[6]:


# Load training data
training = spark.read.format("libsvm").load("sample_linear_regression_data.txt")


# In[8]:


training.show()



# In[9]:


# These are the default values for the featuresCol, labelCol, predictionCol
lr = LinearRegression(featuresCol='features', labelCol='label', predictionCol='prediction')




# In[10]:


# Fit the model
lrModel = lr.fit(training)


# In[16]:


# Print the coefficients and intercept for linear regression
print("Coefficients: {}".format(str(lrModel.coefficients))) # For each feature...
print('\n')
print("Intercept:{}".format(str(lrModel.intercept)))


# There is a summary attribute that contains even more info!

# In[17]:


# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary


# Lots of info, here are a few examples:

# In[36]:


trainingSummary.residuals.show()
print("RMSE: {}".format(trainingSummary.rootMeanSquaredError))
print("r2: {}".format(trainingSummary.r2))


# ## Train/Test Splits

# In[19]:


all_data = spark.read.format("libsvm").load("sample_linear_regression_data.txt")


# In[21]:


# Pass in the split between training/test as a list.
# No correct, but generally 70/30 or 60/40 splits are used. 
# Depending on how much data you have and how unbalanced it is.
train_data,test_data = all_data.randomSplit([0.7,0.3])


# In[27]:


train_data.show()


# In[29]:


test_data.show()


# In[30]:


unlabeled_data = test_data.select('features')


# In[31]:


unlabeled_data.show()


# Now we only train on the train_data

# In[34]:


correct_model = lr.fit(train_data)


# Now we can directly get a .summary object using the evaluate method:

# In[35]:


test_results = correct_model.evaluate(test_data)


# In[40]:


test_results.residuals.show()
print("RMSE: {}".format(test_results.rootMeanSquaredError))


# In[41]:


predictions = correct_model.transform(unlabeled_data)


# In[43]:


predictions.show()

