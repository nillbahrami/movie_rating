from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')

class WhatMovieToWatch:

    def __init__(self,
                 train_dataset,
                 test_dataset):

        self.train = train_dataset
        self.test = test_dataset[['ID', 'Review']]

        self._preprocess()


    def _preprocess(self):

        self._make_x_y()

        self.cleaned_reviews = [self._cleaning_pipeline(review) for review in self.x]
        print(len(self.cleaned_reviews))

        self.test_df = self.test['Review']

        self.cleaned_test_rev = [self._cleaning_pipeline(review) for review in self.test_df]
        print(len(self.cleaned_test_rev))

        self.cv = CountVectorizer(ngram_range = (1, 3))

        self.x_train_vect = self.cv.fit_transform(self.cleaned_reviews)
        print(self.x_train_vect.shape)

        self.x_test_vect = self.cv.transform(self.cleaned_test_rev)
        print(self.x_test_vect.shape)

        print(self.cv.vocabulary_)

    def _make_x_y(self):
        self.x = self.train.Review
        self.y = self.train.Rating


    def _stopWords(self):

        stop_words = stopwords.words('english')
        stop_words.remove('not')
        stop_words = set(stop_words)

        stemmer = PorterStemmer()

        return stemmer, stop_words

    def _cleaning_pipeline(self, review):

        stemmer, stop_words = self._stopWords()

        words = word_tokenize(review.lower())
        words = [stemmer.stem(word) for word in words if word not in stop_words and word.isalpha()]

        review = " ".join(words)

        return review


    def train_model(self):
        # you can use any kind of model you like here!

        mnb = MultinomialNB()

        mnb.fit(self.x_train_vect, self.y)

        mnb.score(self.x_train_vect, self.y)
        y_pred = mnb.predict(self.x_test_vect)

        output = pd.DataFrame()
        output['ID'] = self.test['ID']
        output['prediction'] = y_pred

        return output



if __name__ == '__main__':

    train_dataset = pd.read_csv('train.csv')
    test_dataset = pd.read_csv('test.csv')

    netflix_and_chill_movies = WhatMovieToWatch(train_dataset = train_dataset,
                                                test_dataset = test_dataset)
    picked_movies = netflix_and_chill_movies.train_model()
    picked_movies.to_csv('./picked_movies.csv', index = False)
