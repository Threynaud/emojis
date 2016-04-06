import tweepy
import json
import cnfg
import pandas as pd
import nltk
import string
import emoji
import re
import gensim
import logging
import numpy as np

from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from gensim.models import word2vec

from functools import lru_cache
from pymongo import MongoClient
from requests_oauthlib import OAuth1

cachedStopWords = stopwords.words("english")

emoji_re = emoji.get_emoji_regexp()

# Streaming


def authentify(config_file):
    config = cnfg.load(".twitter_config")

    auth = tweepy.OAuthHandler(config["consumer_key"],
                               config["consumer_secret"])

    auth.set_access_token(config["access_token"],
                          config["access_token_secret"])
    return auth


def create_listener(collection_name):

    class StreamListener(tweepy.StreamListener):
        """
        tweepy.StreamListener is a class provided by tweepy used to access
        the Twitter Streaming API. It allows us to retrieve tweets in real time.
        We store them in emojis database.
        """

        def on_connect(self):
            """Called when the connection is made"""
            print("You're connected to the streaming server.")

        def on_error(self, status_code):
            """This is called when an error occurs"""
            print('Error: ' + repr(status_code))
            return False

        def on_data(self, data):
            """This will be called each time we receive stream data"""
            client = MongoClient('localhost', 27017)

            # Use unhappy database
            db = client.emojis

            # Decode JSON
            datajson = json.loads(data)

            # We only want to store tweets in English
            if "lang" in datajson and datajson["lang"] == "en":
                # Store tweet info into the cooltweets collection if not
                # retweeted.
                if "retweeted" in datajson and not datajson["retweeted"] and 'RT @' not in datajson['text']:
                    db[collection_name].insert(datajson)

    l = StreamListener(api=tweepy.API(wait_on_rate_limit=True))

    return l


def stream_emojis(track, listener, auth):
    l = listener
    streamer = tweepy.Stream(auth=auth, listener=l)

    streamer.filter(track=track)

# Cleaning


def mongo_to_df(db):
    """ Import database db (str) of emojis into a pandas dataframe """
    client = MongoClient()
    emojis = client[db]
    collections = emojis.collection_names()
    df = pd.DataFrame(
        list(emojis[collections[0]].find({}, {"text": 1, "_id": 0})))

    for col in collections[1:]:
        df1 = pd.DataFrame(list(emojis[col].find({}, {"text": 1, "_id": 0})))
        df = df.append(df1, ignore_index=True, verify_integrity=True)

    return df


def df_to_clean_list(df):
    tweets = df['text'].drop_duplicates().values.tolist()

    tweets = [x for x in tweets if not x.startswith('RT')]

    tweets = [i.replace("/n", "")
              for i in tweets if not ('http://' in i or 'https://' in i)]

    return tweets


def init_lemmatizer(cachesize):
    wnl = WordNetLemmatizer()
    lemmatizer = lru_cache(maxsize=cachesize)(wnl.lemmatize)
    return lemmatizer


def clean_tweet(tweet, lemmatizer):
    """ Remove punctuation and lemmatize """
    lowers = tweet.lower()
    # remove the punctuation
    transtable = str.maketrans('', '', string.punctuation)
    no_punctuation = lowers.translate(transtable)

    # tweet = ' '.join([lemmatizer(word) for word in no_punctuation.split() if word not in cachedStopWords])
    tweet = ' '.join([lemmatizer(word) for word in no_punctuation.split()])
    tweet = " ".join(re.findall("(\w+|[^\w ]+)", tweet))
    return tweet


def count_emojis(tweets, nb):
    e = emoji.get_emoji_regexp()

    emojis = []
    for x in tweets:
        match = e.search(x)
        if match:
            emojis.append(match.group())

    dfe = pd.DataFrame(emojis, columns=['text'])
    return dfe

def is_emoji(word):
    if emoji_re.search(word) is not None and len(word) == 1:
        return True
    else:
        return False

# Model


def apply_w2v(tweets, num_features, min_word_count, num_workers, context, downsampling):

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print ("Training model...")

    wt = [list(x.split()) for x in tweets]

    model = word2vec.Word2Vec(wt, workers=num_workers,
                              size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling, iter=9)

    model.init_sims(replace=True)

    print("Done!")

    return model
