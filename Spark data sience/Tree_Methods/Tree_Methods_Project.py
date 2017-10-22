
# coding: utf-8

# # Tree Methods Project

# You've been hired by a dog food company to try to predict why some batches of their dog food are spoiling much quicker than intended! Unfortunately this Dog Food company hasn't upgraded to the latest machinery, meaning that the amounts of the five preservative chemicals they are using can vary a lot, but which is the chemical that has the strongest effect? The dog food company first mixes up a batch of preservative that contains 4 different preservative chemicals (A,B,C,D) and then is completed with a "filler" chemical. The food scientists beelive one of the A,B,C, or D preservatives is causing the problem, but need your help to figure out which one!
# Use Machine Learning with RF to find out which parameter had the most predicitive power, thus finding out which chemical causes the early spoiling! So create a model and then find out how you can decide which chemical is the problem!
# 
# * Pres_A : Percentage of preservative A in the mix
# * Pres_B : Percentage of preservative B in the mix
# * Pres_C : Percentage of preservative C in the mix
# * Pres_D : Percentage of preservative D in the mix
# * Spoiled: Label indicating whether or not the dog food batch was spoiled.
# ___
# 
# **Think carefully about what this problem is really asking you to solve. While we will use Machine Learning to solve this, it won't be with your typical train/test split workflow. 

# In[46]:


#Tree methods Example
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('dogfood').getOrCreate()


# In[47]:


# Load training data
data = spark.read.csv('dog_food.csv',inferSchema=True,header=True)


# In[48]:


data.printSchema()


# In[49]:


data.head()


# In[50]:


data.describe().show()


# In[51]:


# Import VectorAssembler and Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# In[52]:


data.columns


# In[53]:


assembler = VectorAssembler(inputCols=['A', 'B', 'C', 'D'],outputCol="features")


# In[54]:


output = assembler.transform(data)


# In[55]:


from pyspark.ml.classification import RandomForestClassifier,DecisionTreeClassifier


# In[56]:


rfc = DecisionTreeClassifier(labelCol='Spoiled',featuresCol='features')


# In[57]:


output.printSchema()


# In[58]:


final_data = output.select('features','Spoiled')
final_data.head()


# In[59]:


rfc_model = rfc.fit(final_data)


# In[60]:


rfc_model.featureImportances



