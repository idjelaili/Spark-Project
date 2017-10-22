
# coding: utf-8

# # Clustering Project
# 
# A large technology firm needs your help, they've been hacked! Luckily their forensic engineers have grabbed valuable data
# about the hacks, including information like session time,locations, wpm typing speed, etc. 
#The forensic engineer relates to you what she has been able to figure out so far, she has been able to grab meta data of each session that
# the hackers used to connect to their servers. These are the features of the data:
# 
# * 'Session_Connection_Time': How long the session lasted in minutes
# * 'Bytes Transferred': Number of MB transferred during session
# * 'Kali_Trace_Used': Indicates if the hacker was using Kali Linux
# * 'Servers_Corrupted': Number of server corrupted during the attack
# * 'Pages_Corrupted': Number of pages illegally accessed
# * 'Location': Location attack came from (Probably useless because the hackers used VPNs)
# * 'WPM_Typing_Speed': Their estimated typing speed based on session logs.
# 
# 
# The technology firm has 3 potential hackers that perpetrated the attack. Their certain of the first two hackers but they aren't very sure if the third
# hacker was involved or not. They have requested your help! Can you help figure out whether or not the third suspect had anything to do with the attacks, 
#or was it just two hackers? It's probably not possible to know for sure, but maybe what you've just learned about Clustering can help!
# 
# **One last key fact, the forensic engineer knows that the hackers trade off attacks. Meaning they should each have roughly the same amount of attacks. 
#For example if there were 100 total attacks, then in a 2 hacker situation each should have about 50 hacks, in a three hacker situation each would
# have about 33 hacks. The engineer believes this is the key element to solving this, but doesn't know how to distinguish this unlabeled data into groups of hackers.**

# In[173]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('hack_find').getOrCreate()


# In[174]:


from pyspark.ml.clustering import KMeans

# Loads data.
dataset = spark.read.csv("hack_data.csv",header=True,inferSchema=True)


# In[175]:


dataset.head()


# In[176]:


dataset.describe().show()


# In[177]:


df.columns


# In[178]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# In[179]:


feat_cols = ['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used',
             'Servers_Corrupted', 'Pages_Corrupted','WPM_Typing_Speed']


# In[180]:


vec_assembler = VectorAssembler(inputCols = feat_cols, outputCol='features')


# In[181]:


final_data = vec_assembler.transform(dataset)


# In[182]:


from pyspark.ml.feature import StandardScaler


# In[183]:


scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)


# In[184]:


# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(final_data)


# In[185]:


# Normalize each feature to have unit standard deviation.
cluster_final_data = scalerModel.transform(final_data)


# ** Time to find out whether its 2 or 3! **

# In[187]:


kmeans3 = KMeans(featuresCol='scaledFeatures',k=3)
kmeans2 = KMeans(featuresCol='scaledFeatures',k=2)


# In[188]:


model_k3 = kmeans3.fit(cluster_final_data)
model_k2 = kmeans2.fit(cluster_final_data)


# In[189]:


wssse_k3 = model_k3.computeCost(cluster_final_data)
wssse_k2 = model_k2.computeCost(cluster_final_data)


# In[197]:


print("With K=3")
print("Within Set Sum of Squared Errors = " + str(wssse_k3))
print('--'*30)
print("With K=2")
print("Within Set Sum of Squared Errors = " + str(wssse_k2))



# In[204]:


for k in range(2,9):
    kmeans = KMeans(featuresCol='scaledFeatures',k=k)
    model = kmeans.fit(cluster_final_data)
    wssse = model.computeCost(cluster_final_data)
    print("With K={}".format(k))
    print("Within Set Sum of Squared Errors = " + str(wssse))
    print('--'*30)





# In[205]:


model_k3.transform(cluster_final_data).groupBy('prediction').count().show()


# In[206]:


model_k2.transform(cluster_final_data).groupBy('prediction').count().show()



