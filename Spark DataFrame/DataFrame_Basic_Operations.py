
# coding: utf-8

# # Basic Operations


# In[1]:


from pyspark.sql import SparkSession


# In[2]:


# May take awhile locally
spark = SparkSession.builder.appName("Operations").getOrCreate()


# In[26]:


# Let Spark know about the header and infer the Schema types!
df = spark.read.csv('appl_stock.csv',inferSchema=True,header=True)


# In[28]:


df.printSchema()


# ## Filtering Data
# 

# In[31]:


# Using SQL
df.filter("Close<500").show()


# In[35]:


# Using SQL with .select()
df.filter("Close<500").select('Open').show()


# In[36]:


# Using SQL with .select()
df.filter("Close<500").select(['Open','Close']).show()


# 
# Let's see some examples:

# In[38]:


df.filter(df["Close"] < 200).show()


# In[39]:


# Will produce an error, make sure to read the error!
df.filter(df["Close"] < 200 and df['Open'] > 200).show()


# In[47]:


df.filter( (df["Close"] < 200) & (df['Open'] > 200) ).show()


# In[49]:


df.filter( (df["Close"] < 200) | (df['Open'] > 200) ).show()


# In[51]:



df.filter( (df["Close"] < 200) & ~(df['Open'] < 200) ).show()


# In[46]:


df.filter(df["Low"] == 197.16).show()


# In[52]:


# Collecting results as Python objects
df.filter(df["Low"] == 197.16).collect()


# In[53]:


result = df.filter(df["Low"] == 197.16).collect()


# In[62]:


# Note the nested structure returns a nested row object
type(result[0])


# In[65]:


row = result[0]


# Rows can be called to turn into dictionaries

# In[64]:


row.asDict()


# In[59]:


for item in result[0]:
    print(item)



