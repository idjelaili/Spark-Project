
# coding: utf-8

# # Data Transformations
# 


# In[2]:


from pyspark.sql import SparkSession


# In[3]:


spark = SparkSession.builder.appName('data').getOrCreate()


# In[4]:


df = spark.read.csv('fake_customers.csv',inferSchema=True,header=True)


# In[5]:


df.show()


# ## Data Features
# 
# ### StringIndexer
# 

# In[6]:


from pyspark.ml.feature import StringIndexer

df = spark.createDataFrame(
    [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
    ["user_id", "category"])

indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexed = indexer.fit(df).transform(df)
indexed.show()


# The next step would be to encode these categories into "dummy" variables.

# ### VectorIndexer
# 

# In[14]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

dataset = spark.createDataFrame(
    [(0, 18, 1.0, Vectors.dense([0.0, 10.0, 0.5]), 1.0)],
    ["id", "hour", "mobile", "userFeatures", "clicked"])
dataset.show()


# In[15]:


assembler = VectorAssembler(
    inputCols=["hour", "mobile", "userFeatures"],
    outputCol="features")

output = assembler.transform(dataset)
print("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
output.select("features", "clicked").show()



