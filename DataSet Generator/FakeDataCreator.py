
# coding: utf-8

# # Fake Data Creator
# ## Imports and Helper Functions

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from random import uniform 
from faker import Factory,Faker
from collections import defaultdict
from sklearn.preprocessing import normalize,MinMaxScaler
from pandas import to_datetime
from random import randint
get_ipython().magic('matplotlib inline')
fake = Faker()


# In[2]:


def fixer(x1,mu1,std1,mu2):
    '''
    Fixes column values to be more realistic.
    ''' 
    std2 = mu1/4.1
    return ((x1-mu1)/(std1)) * (std2) + mu2


# In[3]:


def realizer(data,minmax,rounder):
    mm = MinMaxScaler((minmax[0],minmax[1]))
    if rounder != '':
        return np.round(mm.fit_transform(data.as_matrix().reshape(-1,1)) , 0)  
    else:
        return np.round(mm.fit_transform(data.as_matrix().reshape(-1,1)) , 2)   


# ## Dictionary of Faker Providers

# In[4]:


date_past = to_datetime('2006-01-01 11:42:52')
date_present = to_datetime('2017-01-01 11:42:52')

fake_codes = {}

fake_codes['phone_number'] = lambda x: fake.phone_number()
fake_codes['name'] = lambda x: fake.name()
fake_codes['catch_phrase'] = lambda x: fake.catch_phrase()
fake_codes['city'] = lambda x: fake.city()
fake_codes['binary'] = lambda x: randint(0,1)
fake_codes['randint1to10'] = lambda x: randint(1,10)
fake_codes['country'] = lambda x: fake.country()
fake_codes['timestamp_this_year'] = lambda x: fake.date_time_this_year()
fake_codes['timestamp'] = lambda x: fake.date_time_between_dates(date_past,date_present)
fake_codes['address'] = lambda x:  fake.address()
fake_codes['lot'] = lambda x:  fake.bothify()
fake_codes['AM or PM'] = lambda x:  fake.am_pm()
fake_codes['browser_info'] = lambda x:  fake.user_agent()
fake_codes['company'] = lambda x:  fake.company()
fake_codes['cc_num'] = lambda x:  fake.credit_card_number()
fake_codes['cc_exp'] = lambda x:  fake.credit_card_expire()
fake_codes['cc_sec_code'] = lambda x:  fake.credit_card_security_code()
fake_codes['cc_provider'] = lambda x:  fake.credit_card_provider()
fake_codes['email'] = lambda x:  fake.email()
fake_codes['job'] = lambda x:  fake.job()
fake_codes['ipv4'] = lambda x:  fake.ipv4()
fake_codes['language'] = lambda x:  fake.language_code()


# _______

# # Fake Classification Tasks
# 
# For classification tasks. The purpose of this function is to return a fake data set that can be used for classification. 
# 
# 
# You pass in a list in the form 
# 
#     [(('col_name','fakercode',(min,max,mu))]

# ______

# In[15]:


def fake_data_classification(dataname='myfakedata',nsamples=1,
                             nclass=2,datacode=[],target_name="target",pct_pos=0.5,std=2):
    """
    INPUT: Takes in a string of the dataname, and a dictionary of column names along with
           what they represent using the Faker() library.
    OUTPUT: Prints the head of the dataframe and also saves it to a csv file.
    """
    print("Generating Classification Data...")
    # Create Data
    data = datasets.make_blobs(n_samples=nsamples, n_features=len(datacode), 
                           centers=nclass, cluster_std=std)

    # Convert to DataFrames with normalized numbers
    features = pd.DataFrame(normalize(data[0])).apply(lambda x: x+1)
    features = pd.DataFrame(data[0])
    
    # Target
    target =  pd.DataFrame(data[1])
    target.columns = [target_name]
    
    print("Running Faker Code")
    for colind,(col_name,code,minmax,rounder) in enumerate(datacode):
        
        # Set Data to be Numerically Realistic
        features[colind] = realizer(features[colind],minmax,rounder)
        # Rename the column as required
        features=features.rename(columns = {colind:col_name})
        
        # Check to see if there is fake data to be generated!
        if code != 'none':
            features[col_name] = features[col_name].apply(fake_codes[code])
        
    ratio = pct_pos*(2/0.5)

    print("Completed Faker Generation")
    print("Saving as "+dataname+".csv")
    final_data = pd.concat([features,target],axis=1)
    
    # Fix issues with 
    num_pos = len(final_data[final_data[target_name]==1])
    positives = final_data[final_data[target_name]==1].sample(np.round(num_pos*ratio))
    negatives = final_data[final_data[target_name]==0]
    
    final_data = pd.concat([positives,negatives])
    final_data.to_csv(dataname+".csv",index=False)
    final_pos = len(final_data[final_data[target_name]==1])
    final_neg = len(final_data[final_data[target_name]==0])
    print("The number of positive points are: {}".format(final_pos))
    print("The number of negative points are: {}".format(final_neg))
    print("Percent of positive points is: {}".format(final_pos/(final_neg+final_pos)))

    
    print("Example of DataFrame Created:")
    print("\n")
    print(final_data.head())


# In[24]:


def multi_fake_data_classification(dataname='myfakedata',nsamples=1,
                             nclass=3,datacode=[],target_name="target",std=2):
    """
    INPUT: Takes in a string of the dataname, and a dictionary of column names along with
           what they represent using the Faker() library.
    OUTPUT: Prints the head of the dataframe and also saves it to a csv file.
    """
    print("Generating Classification Data...")
    # Create Data
    data = datasets.make_blobs(n_samples=nsamples, n_features=len(datacode), 
                           centers=nclass, cluster_std=std)

    # Convert to DataFrames with normalized numbers
    features = pd.DataFrame(normalize(data[0])).apply(lambda x: x+1)
    features = pd.DataFrame(data[0])
    
    # Target
    target =  pd.DataFrame(data[1])
    target.columns = [target_name]
    
    print("Running Faker Code")
    for colind,(col_name,code,minmax,rounder) in enumerate(datacode):
        
        # Set Data to be Numerically Realistic
        features[colind] = realizer(features[colind],minmax,rounder)
        # Rename the column as required
        features=features.rename(columns = {colind:col_name})
        
        # Check to see if there is fake data to be generated!
        if code != 'none':
            features[col_name] = features[col_name].apply(fake_codes[code])
        
    

    print("Completed Faker Generation")
    print("Saving as "+dataname+".csv")
    final_data = pd.concat([features,target],axis=1)
    
    
    print("Example of DataFrame Created:")
    print("\n")
    print(final_data.describe())


# In[21]:


def multi_fake_data_classification(dataname='myfakedata',nsamples=1,nclass=3,datacode=[],target_name="target",std=2):
    

    data = datasets.make_blobs(n_samples=nsamples, n_features=len(datacode), centers=nclass, cluster_std=std)
    return data


# ## Data Sets for clustering

# In[25]:


# Hacker Data Set
base = [
    ("Session_Connection_Time",'none',(1,60),'yes'),
    ("Bytes Transferred",'none',(10,2000),''),
    ("Kali_Trace_Used",'binary',(0,1),''),
    ("Servers_Corrupted",'none',(1,10),''),
    ("Pages_Corrupted",'none',(3,15),'yes'),
    ("Location","country",(0,1),''),
    ("WPM_Typing_Speed",'none',(40,75),'')
]
result = multi_fake_data_classification('hack_data',nsamples=500,nclass=3,datacode=base,
                                  target_name="hack", std=1)


# In[23]:





# In[19]:


pd.read_csv('hack_data.csv').describe()


# ## Creating Classification Sets
# 
# ### Binary Customer Churn
# 
# A marketing agency has many customers that use their service to produce ads for the client/customer websites. They've noticed that they have quite a bit of churn in clients. They basically randomly assign account managers right now, but want you to create a machine learning model that will help predict which customers will churn (stop buying their service) so that they can correctly assign the customers most at risk to churn an account manager. Luckily they have some historical data, can you help them out? Create a classification algorithm that will help classify whether or not a customer churned. Then the company can test this against incoming data for future customers to predict which customers will churn and assign them an account manager.
# 
# Here are the fields and their definitions:
# * Name : Name of the latest contact at Company
# * Age: Customer Age
# * Total_Purchase: Total Ads Purchased
# * Account_Manager: Binary 0=No manager, 1= Account manager assigned
# * Years: Totaly Years as a customer
# * Num_sites: Number of websites that use the service.
# * Onboard_date: Date that the name of the latest contact was onboarded
# * Location: Client HQ Address
# * Company: Name of Client Company

# In[172]:


base = [
    ('Names','name',(0,1),''),
    ("Age",'none',(22,65),'yes'),
    ("Total_Purchase",'none',(100,20000),''),
    ("Account_Manager",'binary',(0,1),''),
    ("Years",'none',(1,10),''),
    ("Num_Sites",'none',(3,15),'yes'),
    ("Onboard_date",'timestamp',(0,1),''),
    ("Location","address",(0,1),''),
    ("Company","company",(0,1),'')
]
result = fake_data_classification('new_customers',nsamples=10,nclass=2,datacode=base,
                                  target_name="Churn",pct_pos=0.05,std=6.2)


# ### Check with X,y split
# *Don't edit this!*

# In[134]:


df = pd.read_csv("customer_churn.csv")
y = df['Churn']
X = df[['Age',"Total_Purchase","Account_Manager",'Years',"Num_Sites"]]
print(classification_report(y_test,predictions))


# ## Dog Food
# 
# You've been hired by a dog food company to try to predict why some batches of their dog food are spoiling much quicker than intended! Unfortunately this Dog Food company hasn't upgraded to the latest machinery, meaning that the amounts of the five preservative chemicals they are using can vary a lot, but which is the chemical that has the strongest effect? The dog food company first mixes up a batch of preservative that contains 4 different preservative chemicals (A,B,C,D) and then is completed with a "filler" chemical. The food scientists beelive one of the A,B,C, or D preservatives is causing the problem, but need your help to figure out which one!
# 
# Use Machine Learning with RF to find out which parameter had the most predicitive power, thus finding out which chemical causes the early spoiling! So create a model and then find out how you can decide which chemical is the problem!
# 
# * Pres_A : Percentage of preservative A in the mix
# * Pres_B : Percentage of preservative B in the mix
# * Pres_C : Percentage of preservative C in the mix
# * Pres_D : Percentage of preservative D in the mix
# * Other: Filler chemical that is not a preservative

# In[140]:


base = [
     ('A','randint1to10',(0,1),''),
    ('B','randint1to10',(0,1),''),
    ('C','none',(5,15),'yes'),
    ('D','randint1to10',(0,1),''),
]

result = fake_data_classification('dog_food2',nsamples=700,nclass=2,datacode=base,
                                  target_name="Spoiled",pct_pos=0.1,std=3.2)


# In[146]:


df = pd.read_csv("dog_food.csv")
y = df['Spoiled']
X = df[['A',"B","C",'D']]
print(classification_report(y_test,predictions))
print(df.columns)
print(rfc.feature_importances_)


# ### Test Validity with RF

# In[142]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[143]:


# X = df.drop('Class'.as_matrix()
# y = np.ravel(target.as_matrix())
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33)
rfc = RandomForestClassifier(200)
rfc.fit(X_train,y_train)   
predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))


# ## Cancer Detection
# 
# Just use breast cancer data

# ## Possible regression task with Facebook Data!

# In[149]:


df = pd.read_csv('DataSets/dataset_Facebook.csv',sep=';')


# _______
# 
# ______

# # Regression Tasks

# In[70]:


def fake_data_regression(dataname='myfakedata',nsamples=1,datacode=[],target_name="target"):
    """
    INPUT: Takes in a string of the dataname, and a dictionary of column names along with
           what they represent using the Faker() library.
    OUTPUT: Prints the head of the dataframe and also saves it to a csv file.
    """
    print("Generating Classification Data...")
    std = uniform(1,2)
    print("Random std value was: "+str(std))
    # Create Data
    data = datasets.make_regression(n_samples=nsamples, n_features=len(datacode), 
                                    n_informative=len(datacode), n_targets=1,noise=0.2)
    
    # Convert to DataFrames with normalized numbers
    features = pd.DataFrame(normalize(data[0])).apply(lambda x: x+1)
    features = pd.DataFrame(data[0])
    
    # Target
    target =  pd.DataFrame(data[1])
    target.columns = [target_name]
    
    print("Running Faker Code")
    for colind,(col_name,code,minmax) in enumerate(datacode):
        
        # Set Data to be Numerically Realistic
        features[colind] = realizer(features[colind],minmax)
        # Rename the column as required
        features=features.rename(columns = {colind:col_name})
        
        # Check to see if there is fake data to be generated!
        if code != 'none':
            features[col_name] = features[col_name].apply(fake_codes[code])
        
        
    print("Completed Faker Generation")
    print("Saving as "+dataname+".csv")
    final_data = pd.concat([features,target],axis=1)
    final_data.to_csv(dataname+".csv")
    print("Example of DataFrame Created:")
    print("\n")
    print(final_data.head())
    return final_data
                           


# In[ ]:


fake_data_regression()

