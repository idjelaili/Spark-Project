
# coding: utf-8

# # GroupBy and Aggregate Functions

# In[1]:


from pyspark.sql import SparkSession


# In[2]:



spark = SparkSession.builder.appName("groupbyagg").getOrCreate()


# Read in the customer sales data

# In[3]:


df = spark.read.csv('sales_info.csv',inferSchema=True,header=True)


# In[4]:


df.printSchema()


# In[8]:


df.show()


# Let's group together by company!

# In[9]:


df.groupBy("Company")


# This returns a GroupedData object, off of which you can all various methods

# In[10]:


# Mean
df.groupBy("Company").mean().show()


# In[11]:


# Count
df.groupBy("Company").count().show()


# In[12]:


# Max
df.groupBy("Company").max().show()


# In[13]:


# Min
df.groupBy("Company").min().show()


# In[15]:


# Sum
df.groupBy("Company").sum().show()




# In[18]:


# Max sales across everything
df.agg({'Sales':'max'}).show()


# In[22]:


# Could have done this on the group by object as well:


# In[23]:


grouped = df.groupBy("Company")


# In[25]:


grouped.agg({"Sales":'max'}).show()


# ## Functions

# In[36]:


from pyspark.sql.functions import countDistinct, avg,stddev


# In[29]:


df.select(countDistinct("Sales")).show()


# Often you will want to change the name, use the .alias() method for this:

# In[31]:


df.select(countDistinct("Sales").alias("Distinct Sales")).show()


# In[35]:


df.select(avg('Sales')).show()


# In[38]:


df.select(stddev("Sales")).show()


# That is a lot of precision for digits! Let's use the format_number to fix that!

# In[39]:


from pyspark.sql.functions import format_number


# In[40]:


sales_std = df.select(stddev("Sales").alias('std'))


# In[41]:


sales_std.show()


# In[42]:


# format_number("col_name",decimal places)
sales_std.select(format_number('std',2)).show()


# ## Order By
# 
# You can easily sort with the orderBy method:

# In[43]:


# OrderBy
# Ascending
df.orderBy("Sales").show()


# In[47]:


# Descending call off the column itself.
df.orderBy(df["Sales"].desc()).show()

