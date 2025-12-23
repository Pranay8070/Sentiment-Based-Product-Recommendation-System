from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

# ML libraries (needed for unpickling)
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xg
import pickle
import pandas as pd
import numpy as np
import re
import string
import os
import nltk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
nltk.data.path.append(NLTK_DATA_DIR)

class SentimentRecommenderModel:
    ROOT_PATH = os.path.join(BASE_DIR, "pickle")
    VECTORIZER = "pickle_tfidf-vectorizer.pkl"
    MODEL_NAME = "pickle_sentiment-classification-xg-boost-model.pkl"
    RECOMMENDER = "pickle_user_final_rating.pkl"
    CLEANED_DATA = "pickle_cleaned-data.pkl"

    def load_pickle(self, filename):
        file_path = os.path.join(self.ROOT_PATH, filename)
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def __init__(self):
        self.vectorizer = self.load_pickle(self.VECTORIZER)
        self.model = self.load_pickle(self.MODEL_NAME)
        self.user_final_rating = self.load_pickle(self.RECOMMENDER)
        self.data = pd.read_csv(os.path.join(BASE_DIR, "data", "sample30.csv"))
        self.cleaned_data = self.load_pickle(self.CLEANED_DATA)

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    """function to get the top product 20 recommendations for the user"""

    def getRecommendationByUser(self, user):
        recommedations = []
        return list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)  
    
    """function to filter the product recommendations using the sentiment model and get the top 5 recommendations"""
    

    def getSentimentRecommendations(self, user):
        if (user in self.user_final_rating.index):
            # get the product recommedation using the trained ML model
            recommendations = list(
                self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index) 
            
            # filter the reviews for the recommended products
            filtered_data = self.cleaned_data[self.cleaned_data.id.isin(recommendations)].copy()
            # preprocess the text before tranforming and predicting
            #filtered_data["reviews_text_cleaned"] = filtered_data["reviews_text"].apply(lambda x: self.preprocess_text(x))
            # transfor the input data using saved tf-idf vectorizer

            X = self.vectorizer.transform(
                filtered_data["reviews_text_cleaned"].values.astype(str))
            # predict the sentiment using the saved model
            filtered_data = filtered_data.copy()
            filtered_data["predicted_sentiment"] = self.model.predict(X)
            # calculate the positive sentiment percentage for each product
            temp = filtered_data[['id', 'predicted_sentiment']]
            temp_grouped = temp.groupby('id', as_index=False).count()
            temp_grouped["pos_review_count"] = temp_grouped.id.apply(lambda x: temp[(
                temp.id == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count())
            
            temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
            temp_grouped['pos_sentiment_percent'] = np.round(
                temp_grouped["pos_review_count"]/temp_grouped["total_review_count"]*100, 2)
            sorted_products = temp_grouped.sort_values(
                'pos_sentiment_percent', ascending=False)[0:5]
            
            return pd.merge(self.data, sorted_products, on="id")[["name", "brand", "manufacturer", "pos_sentiment_percent"]].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])
        else:
            print(f"User name {user} doesn't exist")
            # return an empty DataFrame instead of a list
            return pd.DataFrame(columns=["name", "brand", "manufacturer", "pos_sentiment_percent"])

    """function to classify the sentiment to 1/0 - positive or negative - using the trained ML model"""
    def classifySentiment(self, review_text):
        # preprocess the text before tranforming and predicting
        review_text = self.preprocess_text(review_text)
        # transfor the input data using saved tf-idf vectorizer
        X = self.vectorizer.transform([review_text])
        # predict the sentiment using the saved model
        y_pred = self.model.predict(X)
        return y_pred
    
    def preprocess_text(self, text):

        # cleaning the review text (lower, removing punctuation, numericals, whitespaces)
        text = text.lower().strip()
        text = re.sub("\[\s*\w*\s*\]", "", text)
        dictionary = "abc".maketrans('', '', string.punctuation)
        text = text.translate(dictionary)
        text = re.sub("\S*\d\S*", "", text)

        # remove stop-words and convert it to lemma
        text = self.lemma_text(text)
        return text

    """function to get the pos tag to derive the lemma form"""

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    """function to remove the stop words from the text"""

    def remove_stopword(self, text):
        words = [word for word in text.split() if word.isalpha()
                 and word not in self.stop_words]
        return " ".join(words)

    """function to derive the base lemma form of the text using the pos tag"""

    def lemma_text(self, text):
        word_pos_tags = nltk.pos_tag(word_tokenize(
            self.remove_stopword(text)))  # Get position tags
        # Map the position tag and lemmatize the word/token
        words = [self.lemmatizer.lemmatize(tag[0], self.get_wordnet_pos(
            tag[1])) for idx, tag in enumerate(word_pos_tags)]
        return " ".join(words)
