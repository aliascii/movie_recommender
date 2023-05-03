from flask import Flask, render_template, request
from recommender import random_recommender, MOVIES

app = Flask(__name__)

@app.route('/')
def homepage(): 
    return render_template('home.html')

@app.route('/results') #Decorator to take function's output and giving it back to user in a browser (rendering)
def recommender():
    top_two = random_recommender()
    return f'{top_two}'

# @app.route('/results/')
# def recommender():
#     print(request.args)
#     return f'Here are some recommended movies: {random_recommender()}'

# @app.route('/cos_sim_results')
# def recommender():
#     user_query = request.args.to_dict()
#     user_query = {key:int(value) for key, value in user_query.items()}
#     top_two = co_sim()
#     return render_template('results.html', movies = top_two)

# @app.route('/NMF_results')
# def recommender():
#     user_query = request.args.to_dict()
#     user_query = {key:int(value) for key, value in user_query.items()}
#     top_two = NMF()
#     return render_template('results.html', movies = top_two)

if __name__ == '__main__':
    app.run(port=5001, debug=True)


#'q=jungle+book&oq=jungle+book&aqs=chrome.0.0i271j46i340i433i512l3j0i512j46i512j0i512j46i512j0i512l2.2001j0j7&sourceid=chrome&ie=UTF-8'