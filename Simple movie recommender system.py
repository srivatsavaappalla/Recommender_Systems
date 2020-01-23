
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np


# In[ ]:


pd.read_csv()


# In[37]:


u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code'] 
users = pd.read_csv('u.user', sep='|', names=u_cols,encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('u.data', sep='\t', names=r_cols, encoding ='latin-1')

i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'] 
items = pd.read_csv('u.item', sep='|', names=i_cols, encoding='latin-1')


# In[75]:


users.shape


# In[36]:


users.shape
users.head()


# In[38]:


ratings.shape
ratings.head()


# In[22]:


items.shape


# In[23]:


items.head()


# In[24]:


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp'] 
ratings_train = pd.read_csv('ua.base', sep='\t', names=r_cols, encoding='latin-1') 
ratings_test = pd.read_csv('ua.test', sep='\t', names=r_cols, encoding='latin-1') 
ratings_train.shape, ratings_test.shape


# In[66]:


n_users = ratings.user_id.unique().shape[0]


# In[67]:


n_users


# In[70]:


n_items = ratings.movie_id.unique().shape[0]
n_items


# In[78]:


data_matrix = np.zeros((n_users, n_items))

for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]


# In[91]:


pd.DataFrame(data=user_prediction).head()
pd.DataFrame(item_prediction).head()


# In[81]:


from sklearn.metrics.pairwise import pairwise_distances 
user_similarity = pairwise_distances(data_matrix, metric='cosine') 
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')


# In[88]:


#for user based cf - prediction(u,i) = sigma(r(u,i)*similarity(u,v))/sigma(sim(u,v)) where u,v are user, i is items
#for item based cf - prediction (u,i) = sigma(R(u,N) * similarity(i,N))/sigma(sim(i,N)) where 

def predict(ratings, similarity, type='user'):    
    if type == 'user':        
        mean_user_rating = ratings.mean(axis=1)               
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])        
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T    
    elif type == 'item':       
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])    
    return pred


# In[89]:


user_prediction = predict(data_matrix, user_similarity, type='user') 
item_prediction = predict(data_matrix, item_similarity, type='item')

