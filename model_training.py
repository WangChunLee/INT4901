# Import libraries
import implicit
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import math
import random
import pickle



'''''''''''''''''''''''''''''''''''''''''''''
---Read data and build user-item matrix---  '
'''''''''''''''''''''''''''''''''''''''''''''

# Read data from CSV to pandas dataframe format
df = pd.read_csv('view_record.csv', header=None)    # Read view record
df.columns = ["user_id", "book_id", "confidence"]

# Set user id and book id as category, so that multiple occurence of user/book id will be counted into same categories
df['user_id'] = df['user_id'].astype("category")
df['book_id'] = df['book_id'].astype("category")


# Properties and functions for easier accessing with original user id
user_cat_index = dict(enumerate(df['user_id'].cat.categories))
item_cat_index = dict(enumerate(df['book_id'].cat.categories))

user_index_keys = list(user_cat_index.keys())
user_index_values = list(user_cat_index.values())
item_index_keys = list(item_cat_index.keys())
book_index_values = list(item_cat_index.values())

def user_id_to_internal_id(user_id):
    return user_index_keys[user_index_values.index(user_id)]

def book_id_to_internal_id(book_id):
    return item_index_keys[book_index_values.index(book_id)]


# Building user-item matrices
item_user_matrix = sparse.csr_matrix((df['confidence'].astype(float), (df['book_id'].cat.codes, df['user_id'].cat.codes)))    # for model training
user_item_matrix = sparse.csr_matrix((df['confidence'].astype(float), (df['user_id'].cat.codes, df['book_id'].cat.codes)))    # for giving recommendations




'''''''''''''''''''''''''''''''''''''''
---Making train set and test set---   '
'''''''''''''''''''''''''''''''''''''''

# define a function for making train and test set, accept ratings matrix as parameter, and a percentage of entries masking

def make_train_and_test(ratings_matrix, mask_percentage = 0.2):
    
    # ===== Making train set ===== #

    train_set = ratings_matrix.copy()   # declare a train set variable for storing original interaction matrix
    
    # Return the indices where interactions exist (confidence > 0)
    interacted_indices = train_set.nonzero()
    interacted_items = list(zip(interacted_indices[0], interacted_indices[1]))

    random.seed(0) # Set the random seed to zero for reproducibility

    # Get the indices to be masked according to the mask_percentage
    mask_num = int(np.ceil(mask_percentage * len(interacted_items)))
    interactions_to_mask = random.sample(interacted_items, mask_num)

    # Set the random chosen matrix indices to 0 (masking)
    mask_item_index = [interaction[0] for interaction in interactions_to_mask]
    mask_user_index = [interaction[1] for interaction in interactions_to_mask]
    train_set[mask_item_index, mask_user_index] = 0

    train_set.eliminate_zeros() # Saving space for sparse matrix
    

    
    # ==== Making test set ===== #
    test_set = ratings_matrix.copy()    # Test set will be our original interaction matrix
    test_set[test_set > 0] = 1          # Data inside test set should be set to binary (to show preference)



    # train set, test set, and a list of modified users
    return train_set, test_set, list(set(mask_user_index))


# call the function to get train set, test set, and a list of modified users
train_set, test_set, modified_users = make_train_and_test(item_user_matrix, mask_percentage = 0.2)




'''''''''''''''''''''''''''''''''''''''''''''''''''
---Model training with the train set using ALS--- '
'''''''''''''''''''''''''''''''''''''''''''''''''''

# Declare an ALS model with a target of 17 latent factors which runs 40 iterations
model = implicit.als.AlternatingLeastSquares(factors=17, iterations=40, calculate_training_loss=True)

# Scale the confidence of the train set
alpha = 20
data = train_set * alpha
data = data.astype('double')

model.fit(data)     # Fit the data to the model


# obtain the user factors and the item factors after model training
user_factors = sparse.csr_matrix(model.user_factors)
item_factors = sparse.csr_matrix(model.item_factors)




### Save the user and item factors to file ###

model = {
    'user_factors': user_factors,
    'item_factors': item_factors,
    'user_item_matrix': user_item_matrix,
    'user_cat_index': user_cat_index,
    'item_cat_index': item_cat_index
}

print("Saving model...")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

print("Saved.")