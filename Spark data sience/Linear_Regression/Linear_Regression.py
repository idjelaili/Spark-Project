
# coding: utf-8

# # Linear Regression 

# In[1]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('lr_example').getOrCreate()


# In[2]:


from pyspark.ml.regression import LinearRegression


# In[3]:


# Use Spark to read in the Ecommerce Customers csv file.
data = spark.read.csv("Ecommerce_Customers.csv",inferSchema=True,header=True)


# In[4]:


# Print the Schema of the DataFrame
data.printSchema()


# In[5]:


data.show()


# In[6]:


data.head()


# In[7]:


for item in data.head():
    print(item)


# ## Setting Up DataFrame for Machine Learning 

# In[8]:


# A few things we need to do before Spark can accept the data!
# It needs to be in the form of two columns
# ("label","features")

# Import VectorAssembler and Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# In[9]:


data.columns


# In[10]:


assembler = VectorAssembler(
    inputCols=["Avg Session Length", "Time on App", 
               "Time on Website",'Length of Membership'],
    outputCol="features")


# In[11]:


output = assembler.transform(data)


# In[12]:


output.select("features").show()


# In[13]:


output.show()


# In[14]:


final_data = output.select("features",'Yearly Amount Spent')


# In[15]:


train_data,test_data = final_data.randomSplit([0.7,0.3])


# In[16]:


train_data.describe().show()


# In[17]:


test_data.describe().show()


# In[18]:


# Create a Linear Regression Model object
lr = LinearRegression(labelCol='Yearly Amount Spent')


# In[19]:


# Fit the model to the data and call this model lrModel
lrModel = lr.fit(train_data,)


# In[20]:


# Print the coefficients and intercept for linear regression
print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))


# In[21]:


test_results = lrModel.evaluate(test_data)


# In[22]:


# Interesting results....
test_results.residuals.show()


# In[23]:


unlabeled_data = test_data.select('features')


# In[24]:


predictions = lrModel.transform(unlabeled_data)


# In[25]:


predictions.show()


# In[26]:


print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))

