
# coding: utf-8

# # Logistic Regression Consulting Project
# 


# ## Binary Customer Churn
# 
# A marketing agency has many customers that use their service to produce ads for the client/customer websites. They've noticed that they have quite a bit of churn in clients. They basically randomly assign account managers right now, but want you to create a machine learning model that will help predict which customers will churn (stop buying their service) so that they can correctly assign the customers most at risk to churn an account manager. Luckily they have some historical data, can you help them out? Create a classification algorithm that will help classify whether or not a customer churned. Then the company can test this against incoming data for future customers to predict which customers will churn and assign them an account manager.
# 
# The data is saved as customer_churn.csv. Here are the fields and their definitions:
# 
#     Name : Name of the latest contact at Company
#     Age: Customer Age
#     Total_Purchase: Total Ads Purchased
#     Account_Manager: Binary 0=No manager, 1= Account manager assigned
#     Years: Totaly Years as a customer
#     Num_sites: Number of websites that use the service.
#     Onboard_date: Date that the name of the latest contact was onboarded
#     Location: Client HQ Address
#     Company: Name of Client Company
#     
# Once you've created the model and evaluated it, test out the model on some new data (you can think of this almost like a hold-out set) that your client has provided, saved under new_customers.csv. The client wants to know which customers are most likely to churn given this data (they don't have the label yet).

# In[1]:


from pyspark.sql import SparkSession


# In[2]:


spark = SparkSession.builder.appName('logregconsult').getOrCreate()


# In[3]:


data = spark.read.csv('customer_churn.csv',inferSchema=True,
                     header=True)


# In[37]:


data.printSchema()




# In[5]:


data.describe().show()


# In[38]:


data.columns


# ### Format for MLlib

# In[7]:


from pyspark.ml.feature import VectorAssembler


# In[8]:


assembler = VectorAssembler(inputCols=['Age',
 'Total_Purchase',
 'Account_Manager',
 'Years',
 'Num_Sites'],outputCol='features')


# In[9]:


output = assembler.transform(data)


# In[39]:


final_data = output.select('features','churn')


# ### Test Train Split

# In[40]:


train_churn,test_churn = final_data.randomSplit([0.7,0.3])


# ### Fit the model

# In[12]:


from pyspark.ml.classification import LogisticRegression


# In[13]:


lr_churn = LogisticRegression(labelCol='churn')


# In[14]:


fitted_churn_model = lr_churn.fit(train_churn)


# In[15]:


training_sum = fitted_churn_model.summary


# In[41]:


training_sum.predictions.describe().show()


# ### Evaluate results
# 
# Let's evaluate the results on the data set we were given (using the test data)

# In[17]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator


# In[18]:


pred_and_labels = fitted_churn_model.evaluate(test_churn)


# In[42]:


pred_and_labels.predictions.show()


# ### Using AUC

# In[24]:


churn_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                           labelCol='churn')


# In[26]:


auc = churn_eval.evaluate(pred_and_labels.predictions)


# In[43]:


auc




# In[28]:


final_lr_model = lr_churn.fit(final_data)


# In[29]:


new_customers = spark.read.csv('new_customers.csv',inferSchema=True,
                              header=True)


# In[30]:


new_customers.printSchema()


# In[31]:


test_new_customers = assembler.transform(new_customers)


# In[32]:


test_new_customers.printSchema()


# In[33]:


final_results = final_lr_model.transform(test_new_customers)


# In[35]:


final_results.select('Company','prediction').show()



