import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import MinMaxScaler
from flask import Flask,jsonify,request


'''''''''''''''''
---Read data--- '
'''''''''''''''''

data = {}

with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

user_factors = data['user_factors']
item_factors = data['item_factors']
user_item_matrix = data['user_item_matrix']
user_cat_index = data['user_cat_index']
item_cat_index = data['item_cat_index']

user_index_keys = list(user_cat_index.keys())
user_index_values = list(user_cat_index.values())
item_index_keys = list(item_cat_index.keys())
book_index_values = list(item_cat_index.values())

def user_id_to_internal_id(user_id):
    return user_index_keys[user_index_values.index(user_id)]

def book_id_to_internal_id(book_id):
    return item_index_keys[book_index_values.index(book_id)]



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
    

    recommendations = []

    # Loop through the preference list, push books and their scores to recommendation list
    for id in id_list:
        recommendations.append({
            'book_id': item_cat_index[id],      # Get the original book id
            'score': recommend_factors[id]      # Score of the book
        })

    return recommendations



'''''''''''''''''''''''''''
---Setting up server---   '
'''''''''''''''''''''''''''

app = Flask(__name__)

# define an API endpoint for fetching the recommendation list, with User ID as parameter
# e.g. calling localhost:5000 with  GET /recommend/user/2, will return the top 10 recommendations for user 2.

@app.route('/recommend/user/<int:num>', methods=['GET'])
def http_recommend(num):
    # call recommend function with user id and return the recommendation list as json
    return jsonify(
        recommend(
            user_id_to_internal_id(num), 
            user_item_matrix, 
            user_factors, item_factors
        )
    )


# Run the server
if(__name__ == '__main__'):
    app.run(debug=True, host='0.0.0.0')