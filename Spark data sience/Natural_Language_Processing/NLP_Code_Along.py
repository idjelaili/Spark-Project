
# coding: utf-8

# # NLP Code Along
# 
# For this code along we will build a spam filter! We'll use the various NLP tools we learned about as well as a new classifier, Naive Bayes.
# 
# We'll use a classic dataset for this - UCI Repository SMS Spam Detection: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

# In[81]:


from pyspark.sql import SparkSession


# In[82]:


spark = SparkSession.builder.appName('nlp').getOrCreate()


# In[83]:


data = spark.read.csv("smsspamcollection/SMSSpamCollection",inferSchema=True,sep='\t')


# In[84]:


data = data.withColumnRenamed('_c0','class').withColumnRenamed('_c1','text')


# In[85]:


data.show()


# ## Clean and Prepare the Data

# ** Create a new length feature: **

# In[86]:


from pyspark.sql.functions import length


# In[87]:


data = data.withColumn('length',length(data['text']))


# In[88]:


data.show()


# In[89]:


# Pretty Clear Difference
data.groupby('class').mean().show()


# ## Feature Transformations

# In[90]:


from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer

tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
ham_spam_to_num = StringIndexer(inputCol='class',outputCol='label')


# In[91]:


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector


# In[92]:


clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')


# ### The Model
# 
# We'll use Naive Bayes

# In[93]:


from pyspark.ml.classification import NaiveBayes


# In[94]:


# Use defaults
nb = NaiveBayes()


# ### Pipeline

# In[95]:


from pyspark.ml import Pipeline


# In[102]:


data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])


# In[103]:


cleaner = data_prep_pipe.fit(data)


# In[104]:


clean_data = cleaner.transform(data)


# ### Training and Evaluation!

# In[105]:


clean_data = clean_data.select(['label','features'])


# In[106]:


clean_data.show()


# In[107]:


(training,testing) = clean_data.randomSplit([0.7,0.3])


# In[108]:


spam_predictor = nb.fit(training)


# In[78]:


data.printSchema()


# In[109]:


test_results = spam_predictor.transform(testing)


# In[110]:


test_results.show()


# In[111]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[114]:


acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting spam was: {}".format(acc))



