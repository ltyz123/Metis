from flask import Flask, render_template, request, url_for, redirect
import gettweets
import pickle
from werkzeug.contrib.cache import SimpleCache

application = Flask(__name__)

@application.route('/', methods=['GET'])
def index():
    tweets = []

    # raw_tweets = gettweets.gettweet('model3')
    # tweets = gettweets.structure_results(raw_tweets)

    # pickle.dump(tweets, open('tweets', 'wb'))
    tweets = pickle.load(open('tweets', 'rb'))

    # Weekly trends graph
    # weekly_trends_url = gettweets.weekly_gragh(tweets)
    # print('weekly trends graph URL: ' + weekly_trends_url)

    # # Maps
    # map_url = gettweets.generate_map(tweets)
    # print('locations map URL: ' + map_url)

    # LDA Model
    lda_html = gettweets.lda_modeling_html(tweets['text'])

    # Get and populate sentiment column in tweets
    sentiment = gettweets.get_sentiment(tweets['text'])
    tweets['sentiment'] = sentiment

    # pie_url = gettweets.get_pie(sentiment)
    # print('pie chart URL: ' + pie_url)

    positive_percent = gettweets.positive_percent(sentiment)

    region_count = gettweets.region_num(tweets['Location'])

    rankings = gettweets.get_top_regions(tweets['Location'])

    scatter_html = gettweets.get_scatter(tweets)
    print(len(scatter_html))

    return render_template('index.html',
        tweets_count=len(tweets),
        weekly_trends_url='https://plot.ly/~sakuralin/272',
        map_url='https://plot.ly/~sakuralin/266',
        pie_url='https://plot.ly/~sakuralin/270',
        positive_percent=positive_percent,
        region_count=region_count,
        rankings=rankings,
        lda_html=lda_html,
        scatter_html=scatter_html
    )

@application.route('/', methods=['POST'])
def get_result():
    return render_template(
    	'index.html'
    )

# run the application.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production application.
    application.debug = True
    application.run()