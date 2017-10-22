
# coding: utf-8

# # Project 
# ## Recommender Systems 

# The whole world seems to be hearing about your new amazing abilities to analyze big data and build useful systems for them! You've just taken up a new contract with a new online food delivery company. This company is trying to differentiate itself by recommending new meals to customers based off of other customers likings.
# 
# Can you build them a recommendation system?
# 
# Your final result should be in the form of a function that can take in a Spark DataFrame of a single customer's ratings for various meals and output their top 3 suggested meals. For example:
# 
# Best of luck!
# 
# ** *Note from Jose: I completely made up this food data, so its likely that the actual recommendations themselves won't make any sense. But you should get a similar output to what I did given the example customer dataframe* **

# In[31]:


import pandas as pd


# In[32]:


df = pd.read_csv('movielens_ratings.csv')


# In[33]:


df.describe().transpose()


# In[34]:


df.corr()


# In[35]:


import numpy as np
df['mealskew'] = df['movieId'].apply(lambda id: np.nan if id > 31 else id)


# In[11]:


df.describe().transpose()


# In[36]:


mealmap = { 2. : "Chicken Curry",   
           3. : "Spicy Chicken Nuggest",   
           5. : "Hamburger",   
           9. : "Taco Surprise",  
           11. : "Meatloaf",  
           12. : "Ceaser Salad",  
           15. : "BBQ Ribs",  
           17. : "Sushi Plate",  
           19. : "Cheesesteak Sandwhich",  
           21. : "Lasagna",  
           23. : "Orange Chicken",
           26. : "Spicy Beef Plate",  
           27. : "Salmon with Mashed Potatoes",  
           28. : "Penne Tomatoe Pasta",  
           29. : "Pork Sliders",  
           30. : "Vietnamese Sandwich",  
           31. : "Chicken Wrap",  
           np.nan: "Cowboy Burger",   
           4. : "Pretzels and Cheese Plate",   
           6. : "Spicy Pork Sliders",  
           13. : "Mandarin Chicken PLate",  
           14. : "Kung Pao Chicken",
           16. : "Fried Rice Plate",  
           8. : "Chicken Chow Mein",  
           10. : "Roasted Eggplant ",  
           18. : "Pepperoni Pizza",  
           22. : "Pulled Pork Plate",   
           0. : "Cheese Pizza",   
           1. : "Burrito",   
           7. : "Nachos",  
           24. : "Chili",  
           20. : "Southwest Salad",  
           25.: "Roast Beef Sandwich"}


# In[37]:


df['meal_name'] = df['mealskew'].map(mealmap)


# In[44]:


df.to_csv('Meal_Info.csv',index=False)


# In[39]:


from pyspark.sql import SparkSession


# In[42]:


spark = SparkSession.builder.appName('recconsulting').getOrCreate()


# In[45]:


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS


# In[46]:


data = spark.read.csv('Meal_Info.csv',inferSchema=True,header=True)


# In[47]:


(training, test) = data.randomSplit([0.8, 0.2])


# In[48]:


# Build the recommendation model using ALS on the training data
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="mealskew", ratingCol="rating")
model = als.fit(training)


# In[ ]:


# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)

predictions.show()

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

