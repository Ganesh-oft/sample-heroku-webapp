import pickle
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import sys
import time

model = None
user_final_rating = None
max_recomms = 20
product_feature_dict = None
initialized = None


def load_product_feature(filename='word_vectorizer.pkl'):
    global product_feature_dict
    print("Enter load_product_feature")
    if product_feature_dict is None:
        start = time.time()
        print("Loading product_name_review from csv")
        product_feature_dict = pickle.load(open(filename,'rb'))
#         print(product_name_review.head())
        print("Loaded product_feature_dict from pkl")
        print(f"Time taken to process load_product_feature {time.time() - start}")
    

def load_recommendations(filename='df.csv'):
    global user_final_rating
    print("Enter load_recommendations")
    if user_final_rating is None:
        start = time.time()
        print("Loading user_final_rating from csv")
        user_final_rating = pd.read_csv(filename)
        user_final_rating.set_index('reviews_username', drop=True, inplace=True)
#         print(user_final_rating.head())
        print(f"Loaded user_final_rating from {filename} in time: {time.time() - start} secs")


def load_model(filename='logit_model.pkl'):
    global model
    start = time.time()
    print("Enter load_model")
    model = pickle.load(open(filename, 'rb'))
    print(f"Model loaded from file {filename} in time: {time.time() - start} secs")
    

def get_model(filename='logit_model.pkl'):
    global model
    if model is None:
        load_model(filename)
    return model


def predict(model, input_data):
    return model.predict(input_data)


def classify(recomms):
    print("Enter classify")
    recomm_classify = {}
    for r in recomms:
        predictions = list(predict(get_model(), product_feature_dict[r]))
        sentiment_score = sum(predictions)/len(predictions)
        recomm_classify[r] = sentiment_score
    return recomm_classify


def sort_by_recomm_score(scored_recomms):
    print("Enter sort_by_recomm_score")
    return list(dict(sorted(scored_recomms.items(), key=lambda v:v[1], reverse=True)).keys())


def generate_recommendations(username):
    global user_final_rating
    print("Enter generate_recommendations")
    return list(user_final_rating.loc[username].sort_values(ascending=False)[0:max_recomms].index)



def format_response(username, sorted_recommendations):
    print("Enter format_response")
    return f"""Top 5 result for user {username} are:<br/><br/>1. {sorted_recommendations[0]}<br/>2. {sorted_recommendations[1]}<br/>3. {sorted_recommendations[2]}<br/>4. {sorted_recommendations[3]}<br/>5. {sorted_recommendations[4]}<br/>"""
        
    
def get_recommendations(username):
    start_full = time.time()
    print("Enter get_recommendations")
    global initialized
    if initialized is None:
        print(f"initialized is {initialized}")
        init()
    recommendations = generate_recommendations(username) 
    recommendations_classified = classify(recommendations) 
    sorted_recommendations = sort_by_recomm_score(recommendations_classified)
    formatted_response = format_response(username, sorted_recommendations)
    print(f"Results are:")
    print(f"{formatted_response}")
    print(f"Time taken to complete request: {time.time() - start_full}")
    return formatted_response


def init():
    global initialized
    print("Initializing")
    load_model()
    load_product_feature()
    load_recommendations()
    initialized = True
    print("Initialized")

    
def main(username):
    print(get_recommendations(username))


if __name__ == "__main__":
    username = "joshua"
    main(username)

