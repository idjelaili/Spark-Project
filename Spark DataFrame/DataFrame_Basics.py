
# coding: utf-8

# # Spark DataFrame Basics

# First we need to start a SparkSession:

# In[1]:


from pyspark.sql import SparkSession


# Then start the SparkSession

# In[2]:


spark = SparkSession.builder.appName("Basics").getOrCreate()




# In[3]:

df = spark.read.json('people.json')


# #### Showing the data

# In[4]:

df.show()


# In[5]:


df.printSchema()


# In[6]:


df.columns


# In[7]:


df.describe()


# In[8]:


from pyspark.sql.types import StructField,StringType,IntegerType,StructType


# In[9]:


data_schema = [StructField("age", IntegerType(), True),StructField("name", StringType(), True)]


# In[10]:


final_struc = StructType(fields=data_schema)


# In[11]:


df = spark.read.json('people.json', schema=final_struc)


# In[12]:


df.printSchema()


# ### Grabbing the data

# In[13]:


df['age']


# In[14]:


type(df['age'])


# In[15]:


df.select('age')


# In[16]:


type(df.select('age'))


# In[17]:


df.select('age').show()


# In[18]:


# Returns list of Row objects
df.head(2)


# Multiple Columns:

# In[19]:


df.select(['age','name'])


# In[20]:


df.select(['age','name']).show()


# ### Creating new columns

# In[21]:


# Adding a new column with a simple copy
df.withColumn('newage',df['age']).show()


# In[22]:


df.show()


# In[23]:


# Simple Rename
df.withColumnRenamed('age','supernewage').show()


# More complicated operations to create new columns

# In[24]:


df.withColumn('doubleage',df['age']*2).show()


# In[25]:


df.withColumn('add_one_age',df['age']+1).show()


# In[26]:


df.withColumn('half_age',df['age']/2).show()


# In[27]:


df.withColumn('half_age',df['age']/2)




# ### Using SQL
# 
# To use SQL queries directly with the dataframe, you will need to register it to a temporary view:

# In[28]:


# Register the DataFrame as a SQL temporary view
df.createOrReplaceTempView("people")


# In[29]:


sql_results = spark.sql("SELECT * FROM people")


# In[30]:


sql_results


# In[31]:


sql_results.show()


# In[32]:


spark.sql("SELECT * FROM people WHERE age=30").show()

