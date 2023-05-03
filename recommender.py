'''Implements functions for making predictions.'''
import random

# ratings = pd.read_csv()

MOVIES = [
    'Avatar',
    'The Great Beauty',
    'Star Wars',
    'Interstellar'
]

def random_recommender():
    random.shuffle(MOVIES)
    top_two = MOVIES[0:2]
    return top_two

# def cos_sim():
#     pass

# def NMF():
