# Import libraries
import implicit
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import math
import random
import matplotlib.pyplot as plt



'''''''''''''''''''''''''''''''''''''''''''''
---Read data and build user-item matrix---  '
'''''''''''''''''''''''''''''''''''''''''''''

# Read data from CSV to pandas dataframe format
books_df = pd.read_csv('book_data.csv', header=None)             # Read book data
books_df.columns = ["book_id", "book_title", "category"]

interactions_df = pd.read_csv('view_record.csv', header=None)    # Read view record
interactions_df.columns = ["user_id", "book_id", "confidence"]

df = pd.merge(interactions_df, books_df, on = 'book_id')    # Merge book data and view record


# Set user id and book id as category, so that multiple occurrence of user/book id will be counted into same categories
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




'''''''''''''''''''''''''''''''''''''''''
---Evaluate the model using AUC score---'
'''''''''''''''''''''''''''''''''''''''''

plt.figure(0).clf()


# function for calculating AUC score, compute AUC score according to the TPR and FPR given by the ROC curve

def auc_score(predictions, test, plot=False):
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    auc = metrics.auc(fpr, tpr)
    if plot:
        plt.plot(fpr,tpr,label="auc="+str(auc))
    return auc

# function for calculating mean AUC score of the model, 

def calc_mean_auc(train_set, test_set, modified_users, user_factors, item_factors):

    auc = []    # list used to store the AUC score of each user who has random entries masked
    
    # Loop through each user who has random entries masked

    for user in modified_users:

        # Get the indices of no interaction made by this user
        user_interaction = train_set[:,user].toarray().reshape(-1)       
        no_interaction = np.where(user_interaction == 0)
        
        # Get predictions by filtering all predicted ratings with the "no interaction" indices
        all_predicted_ratings = user_factors[user,:].dot(item_factors).toarray()
        predictions = all_predicted_ratings[0, no_interaction].reshape(-1)
        
        # Get actual result by filtering all actual ratings with the "no interaction" indices
        all_actual_ratings = test_set[:,user].toarray()
        actual_result = all_actual_ratings[no_interaction, 0].reshape(-1)
        
        # Calculate and append the AUC score the the AUC list
        auc.append(auc_score(predictions, actual_result, user==user_id_to_internal_id(12027) ))

    return float('%.3f'%np.mean(auc))   # return the mean AUC of the list



# call the calculate mean AUC function
auc = calc_mean_auc(train_set, test_set, modified_users, user_factors, item_factors.T)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve and AUC score for user 12027")
plt.legend(loc="lower right")
plt.show()

print("The mean AUC score is: ", auc)





'''''''''''''''''''''''''''''''''
---Recommend items to a user--- '
'''''''''''''''''''''''''''''''''

# Functions for recommending items according to user id

def recommend(user_id, user_item_matrix, user_factors, item_factors, amounts=10):
    
    # Get all the predicted matrix entries by computing the dot product of the current user factors and all the item factors (transposed)
    predicted_ratings = user_factors[user_id,:].dot(item_factors.T).toarray()
    
    # Scale the predicted preference from 0 to 1
    scaler = MinMaxScaler()
    predicted_ratings_scaled = scaler.fit_transform(predicted_ratings.reshape(-1,1))[:,0]


    user_interactions = user_item_matrix[user_id,:].toarray().reshape(-1)
    user_interactions = user_interactions + 1   # Add 1 for the sake of later multiplication
    user_interactions[user_interactions > 1] = 0    # Set interacted items to 0 to avoid later multiplication


    recommend_factors = user_interactions * predicted_ratings_scaled    # multiply user interaction by scaled ratings

    id_list = np.argsort(recommend_factors)[::-1][:amounts]     # sort the factors by predicted preference and get certain amounts
    
    
    # Declare list variables for iteration use
    book_ids = []
    book_titles = []
    categories = []
    scores = []

    # Loop through the preference list
    for id in id_list:
        book_id = item_cat_index[id]    # Get the original book id

        # Append the book data to the lists
        book_ids.append(book_id)
        book_titles.append(df.book_title.loc[df.book_id == book_id].iloc[0])
        categories.append(df.category.loc[df.book_id == book_id].iloc[0])

        scores.append(recommend_factors[id])    # Append the score from the recommender factors


    # Return the recommendation in dataframe format
    recommendations = pd.DataFrame({
        'book_id': book_ids, 
        'book_title': book_titles, 
        'category': categories, 
        'score': scores
    })
    return recommendations



### Getting recommendations

def check_recommendations(user_id):
    # Get the top 30 books the user viewed
    print("=====User ", user_id, " viewed=====")
    viewed_books = books_df[books_df["book_id"].isin(interactions_df[interactions_df["user_id"] == user_id].book_id) ]
    print(viewed_books.head(30))

    # Get recommendations to this user
    print("=====Recommend to user ", user_id, "=====")
    recommendations = recommend(user_id_to_internal_id(user_id), user_item_matrix, user_factors, item_factors)
    print(recommendations)

check_recommendations(12900)  # check recommendations for user 12900

#12027