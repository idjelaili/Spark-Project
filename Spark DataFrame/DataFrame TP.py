
# coding: utf-8


# Let's get some quick practice with your new Spark DataFrame skills, you will be asked some basic questions about
# some stock market data, in this case Walmart Stock from the years 2012-2017.
# This exercise will just ask a bunch of questions,
# 
# For now, just answer the questions and complete the tasks below.

# #### Use the walmart_stock.csv file to Answer and complete the  tasks below!

# #### Start a simple Spark Session

# In[5]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("walmart").getOrCreate()


# #### Load the Walmart Stock CSV File, have Spark infer the data types.

# In[66]:


df = spark.read.csv('walmart_stock.csv',header=True,inferSchema=True)


# #### What are the column names?

# In[67]:


df.columns


# #### What does the Schema look like?

# In[68]:


df.printSchema()


# #### Print out the first 5 columns.

# In[76]:


# Didn't strictly need a for loop, could have just then head()
for row in df.head(5):
    print(row)
    print('\n')


# #### Use describe() to learn about the DataFrame.

# In[77]:


df.describe().show()


# ## Bonus Question!
# #### There are too many decimal places for mean and stddev in the describe() dataframe. Format the numbers to just show up to two decimal places. Pay careful attention to the datatypes that .describe() returns, we didn't cover how to do this exact formatting, but we covered something very similar. [Check this link for a hint](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.Column.cast)
# 


# In[78]:


# Uh oh Strings! 
df.describe().printSchema()


# In[79]:


from pyspark.sql.functions import format_number


# In[80]:


result = df.describe()
result.select(result['summary'],
              format_number(result['Open'].cast('float'),2).alias('Open'),
              format_number(result['High'].cast('float'),2).alias('High'),
              format_number(result['Low'].cast('float'),2).alias('Low'),
              format_number(result['Close'].cast('float'),2).alias('Close'),
              result['Volume'].cast('int').alias('Volume')
             ).show()


# #### Create a new dataframe with a column called HV Ratio that is the ratio of the High Price versus volume of stock traded for a day.

# In[81]:


df2 = df.withColumn("HV Ratio",df["High"]/df["Volume"])#.show()
# df2.show()
df2.select('HV Ratio').show()


# #### What day had the Peak High in Price?

# In[88]:


# Didn't need to really do this much indexing
# Could have just shown the entire row
df.orderBy(df["High"].desc()).head(1)[0][0]


# #### What is the mean of the Close column?

# In[89]:


# Also could have gotten this from describe()
from pyspark.sql.functions import mean
df.select(mean("Close")).show()


# #### What is the max and min of the Volume column?

# In[90]:


# Could have also used describe
from pyspark.sql.functions import max,min


# In[92]:


df.select(max("Volume"),min("Volume")).show()


# #### How many days was the Close lower than 60 dollars?

# In[100]:


df.filter("Close < 60").count()


# In[101]:


df.filter(df['Close'] < 60).count()


# In[102]:


from pyspark.sql.functions import count
result = df.filter(df['Close'] < 60)
result.select(count('Close')).show()


# #### What percentage of the time was the High greater than 80 dollars ?
# #### In other words, (Number of Days High>80)/(Total Days in the dataset)

# In[107]:


# 9.14 percent of the time it was over 80
# Many ways to do this
(df.filter(df["High"]>80).count()*1.0/df.count())*100


# #### What is the Pearson correlation between High and Volume?
# #### [Hint](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrameStatFunctions.corr)

# In[110]:


from pyspark.sql.functions import corr
df.select(corr("High","Volume")).show()


# #### What is the max High per year?

# In[112]:


from pyspark.sql.functions import year
yeardf = df.withColumn("Year",year(df["Date"]))


# In[116]:


max_df = yeardf.groupBy('Year').max()


# In[117]:


# 2015
max_df.select('Year','max(High)').show()


# #### What is the average Close for each Calendar Month?
# #### In other words, across all the years, what is the average Close price for Jan,Feb, Mar, etc... Your result will have a value for each of these months. 

# In[121]:


from pyspark.sql.functions import month
monthdf = df.withColumn("Month",month("Date"))
monthavgs = monthdf.select("Month","Close").groupBy("Month").mean()
monthavgs.select("Month","avg(Close)").orderBy('Month').show()



