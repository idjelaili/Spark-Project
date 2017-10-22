
# coding: utf-8

# # Linear Regression Project

# Congratulations! You've been contracted by Hyundai Heavy Industries to help them build a predictive model for some ships. [Hyundai Heavy Industries](http://www.hyundai.eu/en) is one of the world's largest ship manufacturing companies and builds cruise liners.
# 
# You've been flown to their headquarters in Ulsan, South Korea to help them give accurate estimates of how many crew members a ship will require.
# 
# They are currently building new ships for some customers and want you to create a model and use it to predict how many crew members the ships will need.
# 
# Here is what the data looks like so far:
# 
#     Description: Measurements of ship size, capacity, crew, and age for 158 cruise
#     ships.
# 
# 
#     Variables/Columns
#     Ship Name     1-20
#     Cruise Line   21-40
#     Age (as of 2013)   46-48
#     Tonnage (1000s of tons)   50-56
#     passengers (100s)   58-64
#     Length (100s of feet)  66-72
#     Cabins  (100s)   74-80
#     Passenger Density   82-88
#     Crew  (100s)   90-96
#     
# It is saved in a csv file for you called "cruise_ship_info.csv". Your job is to create a regression model that will help predict how many crew members will be needed for future ships. The client also mentioned that they have found that particular cruise lines will differ in acceptable crew counts, so it is most likely an important feature to include in your analysis! 
# 

# In[1]:


from pyspark.sql import SparkSession


# In[2]:


spark = SparkSession.builder.appName('cruise').getOrCreate()


# In[3]:


df = spark.read.csv('cruise_ship_info.csv',inferSchema=True,header=True)


# In[4]:


df.printSchema()


# In[5]:


df.show()


# In[6]:


df.describe().show()


# ## Dealing with the Cruise_line categorical variable


# In[7]:


df.groupBy('Cruise_line').count().show()


# In[8]:


from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="Cruise_line", outputCol="cruise_cat")
indexed = indexer.fit(df).transform(df)
indexed.head(5)


# In[9]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# In[10]:


indexed.columns


# In[11]:


assembler = VectorAssembler(
  inputCols=['Age',
             'Tonnage',
             'passengers',
             'length',
             'cabins',
             'passenger_density',
             'cruise_cat'],
    outputCol="features")


# In[12]:


output = assembler.transform(indexed)


# In[13]:


output.select("features", "crew").show()


# In[14]:


final_data = output.select("features", "crew")


# In[15]:


train_data,test_data = final_data.randomSplit([0.7,0.3])


# In[16]:


from pyspark.ml.regression import LinearRegression
# Create a Linear Regression Model object
lr = LinearRegression(labelCol='crew')


# In[17]:


# Fit the model to the data and call this model lrModel
lrModel = lr.fit(train_data)


# In[18]:


# Print the coefficients and intercept for linear regression
print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))


# In[19]:


test_results = lrModel.evaluate(test_data)


# In[20]:


print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
print("R2: {}".format(test_results.r2))


# In[21]:


# R2 of 0.86 is pretty good, let's check the data a little closer
from pyspark.sql.functions import corr


# In[22]:


df.select(corr('crew','passengers')).show()


# In[23]:


df.select(corr('crew','cabins')).show()

