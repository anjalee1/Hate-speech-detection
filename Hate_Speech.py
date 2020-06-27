import tweepy

# initialize api instance
consumer_key = '5bZAf2lKwTBu6dVZONz5tSZji'
consumer_secret='eDVqN7vZbEni0Ho4dLc5drP6vMdwPJIHrk4WRGcYL0MyV9YzG0'
access_token = '893669907304292353-8ZgFZNnBemESKRiaBb9lHyRPN3LV7a8'
access_token_secret = 'UoL2ojUPP2q0mYvUeKCqFTmGs9bn8TsTlUSFtolLyeKhB'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

import csv
csvFile = open('result.csv','a')
csvWriter = csv.writer(csvFile)

api = tweepy.API(auth, wait_on_rate_limit=True)
for tweet in tweepy.Cursor(api.search, q="jihadism", lang="en").items(10000):
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])

csvFile.close()


import codecs
import nltk
import re
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

with codecs.open('result.csv','r') as csvfile:
    for tweet in csvfile:

        tokenized_tweets = tknzr.tokenize(tweet)

        with open('result1.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(tokenized_tweets)

filename = 'result1.csv'
file = open(filename, 'rt')
text = file.read()
file.close()

import re
alpha_num_values = re.split(r'\W+', text)


filtered_tokens = [x for x in alpha_num_values if not any(c.isdigit() for c in x)]
#print(filtered_tokens)

lemma = nltk.wordnet.WordNetLemmatizer()
stemmed_words = [lemma.lemmatize(word) for word in filtered_tokens]
#print(stemmed_words)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in stemmed_words if not w in stop_words]
# print(words)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


analyzer = SentimentIntensityAnalyzer()

negative_list = []


for word in words:
    sentiment_dict = analyzer.polarity_scores(word)
    if sentiment_dict['compound'] <= - 0.05:

        negative_list.append(word)
# print(negative_list)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(use_idf=True)

matrix=vectorizer.fit_transform(negative_list)
vectorizer.vocabulary
vector= vectorizer.idf_



df = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names())

from sklearn.cluster import KMeans

true_k = 3
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(df)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(true_k):
    print("Cluster %d:" % i)
    print("list of top words:")
    top_words = []
    for ind in order_centroids[i, :6]:
        top_words.append(terms[ind])
    print(top_words)
    print()
for i in range(true_k):
 possible_topic=input("possible topic for cluster %d:" %i)


