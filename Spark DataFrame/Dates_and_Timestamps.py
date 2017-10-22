
# coding: utf-8

# # Dates and Timestamps
# 

# In[1]:


from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("dates").getOrCreate()


# In[3]:


df = spark.read.csv("appl_stock.csv",header=True,inferSchema=True)


# In[4]:


df.show()




# In[44]:


from pyspark.sql.functions import format_number,dayofmonth,hour,dayofyear,month,year,weekofyear,date_format


# In[45]:


df.select(dayofmonth(df['Date'])).show()


# In[46]:


df.select(hour(df['Date'])).show()


# In[8]:


df.select(dayofyear(df['Date'])).show()


# In[11]:


df.select(month(df['Date'])).show()




# In[15]:


df.select(year(df['Date'])).show()


# In[19]:


df.withColumn("Year",year(df['Date'])).show()


# In[29]:


newdf = df.withColumn("Year",year(df['Date']))
newdf.groupBy("Year").mean()[['avg(Year)','avg(Close)']].show()




# In[43]:


result = newdf.groupBy("Year").mean()[['avg(Year)','avg(Close)']]
result = result.withColumnRenamed("avg(Year)","Year")
result = result.select('Year',format_number('avg(Close)',2).alias("Mean Close")).show()



