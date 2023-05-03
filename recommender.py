# 1. Load the necessary libraries
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")


# 2. Load the dataset containing user ratings and movie titles
user_movie = pd.read_csv('./user_movie.csv', index_col=0)

# 3. Preprocess the data and perform feature engineering if necessary
# You may need to normalize the ratings or impute missing values

# 4. Split the data into training and testing sets
# Depending on the size of your dataset, you may need to use cross-validation or other techniques to evaluate your model

# 5. Train an unsupervised learning model
with open('./Factorizer_NMF_2.pkl', 'rb') as file_in:
    fitted_model = pickle.load(file_in)

# 6. Use the trained model to generate movie recommendations for a given user
def recommend_nmf(query, model=fitted_model, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    # define matrices
    USERS = list(user_movie.index)
    MOVIES = list(user_movie.columns)
    Q = fitted_model.components_
    
    # define movie rating average
    movie_mean = user_movie.mean()

    # 1. candidate generation
    user_input = pd.DataFrame(query, index=['new_user'], columns=MOVIES)
    user_input_imputed = user_input.fillna(value=movie_mean)

    # 2. construct new_user-item dataframe given the query
    P_user = model.transform(user_input_imputed)
    P_user = pd.DataFrame(P_user, index=['new_user'])

    # 3. scoring
    R_user_hat = np.dot(P_user, Q)
    R_user_hat = pd.DataFrame(R_user_hat, columns=MOVIES, index=['new_user'])

    # 4. ranking
    R_user_hat_transposed = R_user_hat.T.sort_values(by='new_user', ascending=False)

    # filter out movies already seen by the user
    user_initial_ratings_list = list(query.keys())

    # return the top-k highest rated movie ids or titles
    recommendables = list(R_user_hat_transposed.index)
    recommendations = [movie for movie in recommendables if movie not in user_initial_ratings_list][:k]
    return recommendations



# return the top-k recommended movies to the user
if __name__ =="__main__":
    # example query
    query = {
        "Salem's Lot (2004)": 0,
        "Til There Was You (1997)": 5,
        "Round Midnight (1986)": 5
    }

    # generate recommendations for the given query
    recommendations = recommend_nmf(query, fitted_model)
    k = 10
    top_k_recommendations = recommendations[:k]
    print(top_k_recommendations)
