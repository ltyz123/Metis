from collections import Counter
import csv
from  datetime import time
from datetime import datetime
import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from pprint import pprint
import pyLDAvis
import pyLDAvis.sklearn
import requests
from requests_oauthlib import OAuth1
import string
import sys
import scattertext as st
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import tweepy
from IPython.display import IFrame
from IPython.display import HTML
import jsonpickle
import os
import time
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords


if not nltk.data.find('corpora/stopwords'):
    print("No stopwords found, downlading...")
    nltk.download('stopwords')


# Twitter API
auth = tweepy.AppAuthHandler('1fJN1roPZIDhZZidFDiqkQt78', '7Gm3aAHa3GMnHhbSecgbYogQ96I24RjYiJsu8zF9uJXkXB5kDF')

api = tweepy.API(auth, wait_on_rate_limit=True,
                 wait_on_rate_limit_notify=True)

#Poltly API
py.sign_in('sakuralin', 'WuV2YNZt1hIoiRC9POO3')

if (not api):
    print("Can't Authenticate")
    sys.exit(-1)

#Get tweets
def gettweet(query_topic):
    searchQuery = query_topic  # this is what we're searching for
    maxTweets = 5000 # Some arbitrary large number
    tweetsPerQry = 100  # this is the max the API permits
    fName = [] # We'll store the tweets in a text file.


    # If results from a specific ID onwards are read, set since_id to that ID.
    # else default to no lower limit, go as far back as API allows
    sinceId = None

    # If results only below a specific ID are, set max_id to that ID.
    # else default to no upper limit, start from the most recent tweet matching the search query.
    max_id = -1

    tweetCount = 0
    print("Downloading max {0} tweets".format(maxTweets))
    while tweetCount < maxTweets:
        try:
            if (max_id <= 0):
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry, lang = 'en')
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            since_id=sinceId, lang = 'en')
            else:
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1),lang = 'en')
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1),
                                            since_id=sinceId, lang = 'en')
            if not new_tweets:
                print("No more tweets found")
                break
            for tweet in new_tweets:
                fName.append(tweet)
            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
            break

    return fName

#Get structured tweets
def structure_results(results):
    #Put tweets into a Dataframe
    id_list=[tweet.id for tweet in results]
    data=pd.DataFrame(id_list,columns=['id'])
    data["text"]= [tweet.text.encode('utf-8').decode() for tweet in results]
    data["datetime"]=[tweet.created_at for tweet in results]
    data["Location"]=[tweet.user.location for tweet in results]

    #Get Date
    date = []
    time = []
    for i in data['datetime']:
        date.append(i.date())
        time.append(i.time())
    data = data.sort_values(by='datetime', ascending=True)
    data['weekday'] = data['datetime'].apply(lambda x: datetime.strftime(x, '%A'))
    data['date'] = date
    data['time'] = time

    #Drop dupilicates
    tweet = data.drop_duplicates(subset='text', keep="last")

    #Clean the text
    cleaned_tweets = []
    for doc in tweet['text']:
        doc = re.sub('https:\S*', '', doc)   #remove URL
        doc = re.sub('\\n', ' ', doc)  #remove '/n'
        doc = re.sub('[0-9]', '', doc) #remove numbers
        doc = re.sub('RT', '', doc)  #remove RT
        exclude = set('!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'+'…')
        doc = ''.join(ch for ch in doc if ch not in exclude) #remove punctuation
        cleaned_tweets.append(doc)

    tweet['text'] = cleaned_tweets

    #get removed_stopwords_tweets

    removed_stopwords_tweets = []
    for i in tweet['text']:
        removed = ' '.join([word for word in i.split() if word not in stopwords.words('english')])
        removed = re.sub('#', '', removed)
        removed_stopwords_tweets.append(removed)
    tweet['removed_stopwords_tweets'] = removed_stopwords_tweets

    return tweet

#Get number of tweets
#input:tweet dataframe
def get_number_all(tweet):
    return tweet.shape[0]

#Get weekly trends gragh
#input:tweet dataframe
def weekly_gragh(tweet):
    date_hour = []
    for i in tweet['datetime']:
        date_hour.append(i.strftime('%m/%d %H:00'))
    tweet['date_hour'] = date_hour
    if len(tweet['date'].unique()) > 3:
        weekly_num = tweet.groupby(['date']).size().reset_index(name='number_of_tweets')
        mentioned_times = [go.Scatter(x=weekly_num.date, y= weekly_num.number_of_tweets)]
    else:
        weekly_num = tweet.groupby(['date_hour']).size().reset_index(name='number_of_tweets')
        mentioned_times = [go.Scatter(x=weekly_num.date_hour, y= weekly_num.number_of_tweets)]

    layout = go.Layout(
        autosize=False,
        hovermode='closest',
        width=1000,
        height=400,
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        )
    )

    fig = dict(data=mentioned_times, layout=layout)
    return py.plot(fig, auto_open=False)

#LDA，按之前的代码就可以
#input:tweet['text']
def lda_modeling_html(text):
    n_components = 10

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    ngram_range=(1,5),
                                    max_features=1000,
                                    stop_words='english')
    tf = tf_vectorizer.fit_transform(text)#text = tweet['text']
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)

    vocab = tf_vectorizer.get_feature_names()
    pd = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)

    return pyLDAvis.prepared_data_to_html(pd)

#WordCloud
#input example: wc(tweet['removed_stopwords_tweets'],'White', 'Most Used Words')
# def wc(data,bgcolor,title):
#     plt.figure(figsize = (10,10))
#     wc = WordCloud(background_color = bgcolor, max_words = 1000,  max_font_size = 50)
#     wc.generate(' '.join(data))
#     plt.axis('off')
#     plt.imsave('static/wc.png', wc)

#Get map
#input:tweet['Location']
def clean_location(location):
    x = location.dropna()
    cleaned_location = x[(x !='USA') & (x != 'United States') & (x != '')]
    return cleaned_location

#Get region number
#input:tweet['Location']
def region_num(location):
    return len(clean_location(location).unique())

def get_geocode(cleaned_location):
    # api-endpoint
    location_count = Counter(cleaned_location)
    URL = "https://maps.googleapis.com/maps/api/geocode/json"

    spots=[]
    cleaned_location = cleaned_location.unique()
    counter = 0
    for location in cleaned_location:
        if counter % 200 == 0:
            print('calling google map' + str(counter))

        counter +=1
        # defining a params dict for the parameters to be sent to the API
        PARAMS = {'address':location, 'key':'AIzaSyAcbPOQ8JeRrriVIe8kCq7pkhqTLs-6yqo'}

        # sending get request and saving the response as response object
        r = requests.get(url = URL, params = PARAMS)

        # extracting data in json format
        data = r.json()['results']
        if not data:
            continue

        item = data[0]['geometry']['location']
        item['address'] = location
        item['count'] = location_count[location]
        spots.append(item)
    return spots

#input:spots=get_geocode(clean_location)
def get_map(spots):
    print('got ' + str(len(spots)) + ' spots on map')
    mapbox_access_token = 'pk.eyJ1Ijoic2FrdXJhbGluIiwiYSI6ImNqaW1xM3ZzeTAzN20zcnBkdHBvemtibGwifQ.rB5HXO1FaVftRWviDx9GKg'
    lat = [x['lat'] for x in spots]
    lng = [x['lng'] for x in spots]
    size = [x['count'] for x in spots]
    addresses = [x['address']+' | '+str(x['count']) for x in spots]

    data = [
        go.Scattermapbox(
            lat=lat,
            lon=lng,
            mode='markers',
            marker=dict(
                size=size
            ),
            text=addresses,
        )
    ]

    layout = go.Layout(
        autosize=False,
        width=1000,
        height=600,
        hovermode='closest',
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        ),
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=38.892613,
                lon=-97.360215
            ),
            pitch=0,
            zoom=3
        ),
    )

    fig = dict(data=data, layout=layout)

    return py.plot(fig, auto_open=False)

def generate_map(tweet):
    cleaned_location = clean_location(tweet['Location'])
    spots = get_geocode(cleaned_location)
    return get_map(spots)

#get_region_number
#input: tweet['Location']
def get_top_regions(location):
    cleaned_location = clean_location(location)
    location_count = Counter(cleaned_location)
    return [t for t in location_count.most_common()][:10]

#sentiment
#input:tweet['text']
def get_sentiment(text):
    sentiment = []
    for text in text:
        identification = TextBlob(text).sentiment
        if identification.polarity >0:
            sen = 'positive'
        if identification.polarity <0:
            sen = 'negative'
        if identification.polarity == 0:
            sen = 'neutral'
        sentiment.append(sen)
    return sentiment

#input:sentiment
def get_pie(sentiment):
    labels = ['positive', 'negative', 'neutral']
    values = [sentiment.count('positive'), sentiment.count('negative'), sentiment.count('neutral')]
    colors = ['#81BEF7', '#81F781', '#F7819F']

    trace = go.Pie(labels=labels, values=values,
                   hoverinfo='value', textinfo='label+percent',
                   textfont=dict(size=20),
                   marker=dict(colors=colors,
                               line=dict(color='#FFFFFF', width=5)))

    layout = go.Layout(
        hovermode='closest',
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )

    fig = dict(data=[trace], layout=layout)

    return py.plot(fig, auto_open=False)

#Get percentage of positive tweets
def positive_percent(sentiment):
    return "{:.0%}".format(Counter(sentiment)['positive']/len(sentiment))

#Scattertext
#input:tweet
def get_scatter(tweet):
    scatter_sen = []
    for i in tweet['sentiment']:
        if i == 'negative':
            new_sen = 'negative'
        else:
            new_sen = 'neutral_or_positive'
        scatter_sen.append(new_sen)
    tweet['sen_for_scatter'] = scatter_sen
    scatter_tweet = tweet[['removed_stopwords_tweets','sen_for_scatter']]
    corpus = (st.CorpusFromPandas(scatter_tweet,
                              category_col='sen_for_scatter',
                              text_col='removed_stopwords_tweets',
                              nlp=st.whitespace_nlp_with_sentences)
              .build()
              .get_unigram_corpus()
              .compact(st.ClassPercentageCompactor(term_count=2,
                                                   term_ranker=st.OncePerDocFrequencyRanker)))
    html = st.produce_scattertext_explorer(corpus,
              category='negative',
              category_name='negative',
              not_category_name='neutral_positive',
              width_in_pixels=1000)
    return html
