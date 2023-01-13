from political_nlp import pnlp
import pandas as pd
import plotly.graph_objects as go
import sankey

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
from plotly.subplots import make_subplots

import re as re
import string
import wordcloud as wc
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#nltk.download('stopwords')
import base64

import warnings
warnings.filterwarnings("ignore")


def load_neg_words(filename):
    """
    load list of negative words for negative-ratio sentiment analysis
    :param filename: (str) name of negative words file
    :return: (list) list of negative words
    """

    # open file and read all lines
    f = open(filename, "r")

    # make list of words
    lst = f.readlines()

    # remove new lines for all words
    for i in range(len(lst)):
        lst[i] = lst[i].replace("\n", "")

    # ignore headers
    lst = lst[35:]

    return lst


# make negative words list a global variable
negs = load_neg_words("negative-words.txt")

# saves the topics and their important event info dict as a global variable
topics = {'abortion': [{'x_vline':20, 'x_annotation': 19.6, 'text': 'Roe v Wade Overturned'},
            {'x_vline': 5, 'x_annotation': 4.6, 'text': 'Gonzales v. Planned Parenthood'},
            {'x_vline': 14, 'x_annotation': 13.6, 'text': "Whole Woman's Health v. Hellerstedt"},
            {'x_vline': 18, 'x_annotation': 17.6, 'text': 'June Medical Services v. Russo'},
            {'x_vline':19, 'x_annotation':18.6, 'text':'Texas Six-Week Ban'}],
          'gay marriage': [{'x_vline': 13, 'x_annotation': 12.6, 'text': 'Gay Marriage Legalized Federally'},
            {'x_vline': 1, 'x_annotation': 0.6, 'text': 'Lawrence v. Texas'},
            {'x_vline': 2, 'x_annotation': 1.6, 'text': 'Mass Legalizes Gay Marriage'},
            {'x_vline': 6, 'x_annotation': 5.6, 'text': 'Cali voters approve Proposition 8'},
            {'x_vline': 7, 'x_annotation': 6.6, 'text': 'Matthew Shepard Act'},
            {'x_vline': 8, 'x_annotation': 7.6, 'text': 'Prop 8 deemed unconstitutional'}],
          'marijuana': [{'x_vline': 7, 'x_annotation': 6.6, 'text': 'DOJ Lenient to Medical Marijuana Patients'},
            {'x_vline': 12, 'x_annotation': 11.6, 'text': 'Rohrabacher–Farr amendment'},
            {'x_vline': 16, 'x_annotation': 15.6, 'text': 'CBD leaglized'}],
          'immigration': [{'x_vline':0, 'x_annotation':0.6, 'text': 'Homeland Security Act'}, \
                          {'x_vline':10, 'x_annotation':9.6, 'text':'DACA established'}, \
                          {'x_vline': 15, 'x_annotation':14.6, 'text':'Muslim Travel Ban'}],
          'police brutality': [{'x_vline': 10, 'x_annotation': 9.6, 'text': 'Death of Trayvon Martin'}, \
                          {'x_vline': 11, 'x_annotation': 10.6, 'text': 'BLM Created'}, \
                          {'x_vline': 18, 'x_annotation': 17.6, 'text': 'Death of George Floyd'}]

          }



def load_topic(start, end, topic, negs):
    """
    get sentiment stats and text data for all years of a topic
    :param start: (int) start year
    :param end: (int) end year
    :param topic: (str) topic of interest
    :param negs: (lst) list of negative words
    :return: nyt_passages: (dict) all article text for nyt with year/text for key/val
                nyp_passages: (dict) all article text for nytpwith year/text for key/val
                df: (dataframe) all years and sentiment stats for specified year
    """

    # create dictionaries for nyt and nyp passages
    nyt_passages = {}
    nyp_passages = {}

    # make bool for first iteration of loop
    first = True

    # for each year in specified range
    for i in range(start, end + 1):

        # update which year is being loaded to user
        print("Loading for year:", i)

        lst = []
        lst.append(i)

        # make pnlp object for nyt year of interest and load text
        nyt = pnlp(f"{topic}-nyt/nyt_{topic}_{i}.json", negs)
        nyt.load_text("nyt")

        # make empty string
        nyt_str = ""

        # add all article text to nyt_str
        for k, v in nyt.text.items():
            new = re.sub('[' + string.punctuation + ']', '', v).lower().strip()
            nyt_str += new

        # add nyt_str to nyt_passages dct
        nyt_passages[i] = nyt_str

        # make pnlp object for nyt year of interest and load text
        nyp = pnlp(f"{topic}-nyp/nyp_{topic}_{i}.json", negs)
        nyp.load_text("nyp")

        # make empty string
        nyp_str = ""

        # add all article text to nyt_str
        for k, v in nyp.text.items():
            new = re.sub('[' + string.punctuation + ']', '', v).lower().strip()
            nyp_str += new

        # add nyt_str to nyt_passages dct
        nyp_passages[i] = nyp_str

        # for first iteration of for loop
        if first == True:

            # make a df of first year of interest
            df = pd.DataFrame(
                {'year': [i], "nyt-pos": [nyt.pos], "nyt-neg": [nyt.neg], "nyt-neu": [nyt.neu], "nyp-pos": [nyp.pos],
                 "nyp-neg": [nyp.neg], "nyp-neu": [nyp.neu], "nyt-neg_ratio": [nyt.neg_ratio],
                 "nyp-neg_ratio": [nyp.neg_ratio]})

        # otherwise, make a new df for each year
        else:
            ndf = pd.DataFrame(
                {'year': [i], "nyt-pos": [nyt.pos], "nyt-neg": [nyt.neg], "nyt-neu": [nyt.neu], "nyp-pos": [nyp.pos],
                 "nyp-neg": [nyp.neg], "nyp-neu": [nyp.neu], "nyt-neg_ratio": [nyt.neg_ratio],
                 "nyp-neg_ratio": [nyp.neg_ratio]})

            # concat new df to first year of interest df
            df = pd.concat([df, ndf])

        # no longer = first iteration
        first = False

    # return nyt, nyp passages and stats df
    return nyt_passages, nyp_passages, df


def wordcloud(nyt_texts, nyp_texts):
    """

    :param nyt_texts: (dict) all article text for nyt with year/text for key/val
    :param nyp_texts: (dict) all article text for nytpwith year/text for key/val
    :return:
    """

    # making list of stop words to remove
    stops = list(set(stopwords.words('english')))
    stops.append("s")

    # make strings for all nyt and nyp articles of all years
    nytstring = ''
    nypstring = ''

    # add all texts to strings
    for year in range(2002, 2023):
        nytstring += nyt_texts[year]
        nypstring += nyp_texts[year]

    # split strings
    nyt = nytstring.split()
    nyp = nypstring.split()

    # remove stop words
    for word in stops:
        nyt = [value for value in nyt if value != word]
        nyp = [value for value in nyp if value != word]

    # join words
    nyt = ' '.join(nyt)
    nyp = ' '.join(nyp)

    # make wordcloud object
    cloud = wc.WordCloud()

    # make wordcloud figure for nyt
    nyt_cloud = cloud.generate(nyt)
    fig1 = go.Figure()

    # set oritentation and layout
    fig1.add_trace(go.Image(z=nyt_cloud))
    fig1.update_layout(height=600, width=800, xaxis={'visible': False}, yaxis={'visible': False},
                       title_text='New York Times')

    # make wordcloud figure for nyp
    nyp_cloud = cloud.generate(nyp)
    fig2 = go.Figure()

    # make wordcloud figure for nyt
    fig2.add_trace(go.Image(z=nyp_cloud))
    fig2.update_layout(height=600, width=800, xaxis={'visible': False}, yaxis={'visible': False},
                       title_text='New York Post')

    # return nyt and nyp wordcloud figs
    return fig1, fig2


def parallel(start, end, topic, negs):
    """
    make parallel coordinates plot
    :param df: (dataframe) containing all sentiment score avgs data for each year nyt vs nyp
    :param topic: (str) topic of interest
    :param start: (int) start year of interest
    :param end: (int) end year of interest
    :return: parallel_fig: (fig) parallel coord fig
    """
    # load texts for year of interest
    nyt_texts, nyp_texts, df = load_topic(start, end, topic, negs)

    # make nyt df from only nyt columns
    nyt_df = df[['year', 'nyt-pos', 'nyt-neu', 'nyt-neg']]

    # rename columns
    nyt_df.rename(
        columns={'nyt-pos': 'Positive Sentiment', 'nyt-neu': 'Neutral Sentiment',
                 'nyt-neg': 'Negative Sentiment'}, inplace=True)

    # make journal column
    nyt_df['Journal'] = [0] * len(nyt_df)

    # make nyp df from only nyp columns
    nyp_df = df[['year', 'nyp-pos', 'nyp-neu', 'nyp-neg']]

    # rename columns
    nyp_df.rename(
        columns={'nyp-pos': 'Positive Sentiment', 'nyp-neu': 'Neutral Sentiment',
                 'nyp-neg': 'Negative Sentiment'}, inplace=True)

    # make jounral column
    nyp_df['Journal'] = [1] * len(nyp_df)

    # add together nyt and nyp dfs sorting by year
    parallel_df = pd.concat([nyp_df, nyt_df]).sort_values('year')

    # create vis figure
    parallel_fig = px.parallel_coordinates(parallel_df, color='Journal',
                                           dimensions=['Positive Sentiment', 'Neutral Sentiment', 'Negative Sentiment'],
                                           color_continuous_scale=px.colors.diverging.Tealrose,
                                           color_continuous_midpoint=0.5
                                           )

    return parallel_fig


def sank(num_words, year, topic, negs):
    """
    make sankey vis fig
    :param num_words: (int) number of top nouns for sankey to display for nyt and nyp
    :param year: (int) year of interest to show sankey for
    :param topic: (str) topic of interest
    :param negs: (lst) list of loaded negative words
    :return: fig
    """

    # load noun dictionary with counts for specified year for nyp and nyt
    nyp = pnlp(f"{topic}-nyp/nyp_{topic}_{year}.json", negs)
    nyp.load_text("nyp")
    dct1 = nyp.noun_scores
    nyt = pnlp(f"{topic}-nyt/nyt_{topic}_{year}.json", negs)
    nyt.load_text("nyt")
    dct2 = nyt.noun_scores

    # create list for target, source, and vals
    vlst = []
    wlst = []
    clst = []

    # define for user specified desired number of words to show in sankey
    num = num_words

    # iterate through nyt nouns
    for k, v in dct1.items():

        # if not surpassed user-specified desired number of nouns for sankey
        if num >= 0:

            # append targ, src, and val
            vlst.append(f"New York Post {year} : {topic}")
            wlst.append(k)
            clst.append(v)

        # break when surpassed num_words
        else:
            break
        num -= 1

    # define for user specified desired number of words to show in sankey
    num = num_words

    # iterate through nyp nouns
    for k, v in dct2.items():

        # if not surpassed user-specified desired number of nouns for sankey
        if num >= 0:

            # append targ, src, and val
            vlst.append(f"New York Times {year} : {topic}")
            wlst.append(k)
            clst.append(v)

        # break when surpassed num_words
        else:
            break
        num -= 1

    # make df for src, targ, and vals
    df = pd.DataFrame()
    df["src"] = vlst
    df["targ"] = wlst
    df["vals"] = clst

    # return sankey figure
    return sankey.make_sankey(df, "src", "targ", "vals", year, topic)

#def event_line():

def stacked(df, topic, topic_dict, negs):
    """
    make stacked scatter vis fig
    :param df: (dataframe) df of all sentiment score averages for each year of a topic byt nyt and nyp
    :param topic: (str) topic of interest
    :param topic_dict: (dict) topic of interest as key, numbers for event line as values
    :param negs: (lst) list of all neg words
    :return: fig
    """

    # make list to append year vals (ysed for nyt  and nyp)
    year = []

    # lists for nyt vader sent vals
    nyt_pos_val = []
    nyt_neg_val = []
    nyt_neu_val = []

    # lists for nyp vader sent vals
    nyp_pos_val = []
    nyp_neg_val = []
    nyp_neu_val = []

    # iterate through sentiment stats df
    for i in range(len(df)):
        # append years
        year.append(df.iloc[i]['year'])

        # append pos, neg, neu sentiments for nyt and nyp
        nyt_pos_val.append(df.iloc[i]['nyt-pos'])
        nyp_pos_val.append(df.iloc[i]['nyp-pos'])
        nyt_neg_val.append(df.iloc[i]['nyt-neg'])
        nyp_neg_val.append(df.iloc[i]['nyp-neg'])
        nyt_neu_val.append(df.iloc[i]['nyt-neu'])
        nyp_neu_val.append(df.iloc[i]['nyp-neu'])

    # make subplots
    stackfig = make_subplots(rows=1, cols=2,
                             subplot_titles=("NY Times", "NY Post"))

    # add stacked scatters for nyt vals in row 1, col 1
    # set all layout parameters
    stackfig.add_trace(go.Scatter(
        x=year, y=nyt_pos_val,
        name='pos score',
        mode='lines',
        line=dict(width=0.5, color='green'),
        stackgroup='one',
        groupnorm='percent'  # sets the normalization for the sum of the stackgroup
    ), row=1, col=1)
    stackfig.add_trace(go.Scatter(
        x=year, y=nyt_neg_val,
        name='neg score',
        mode='lines',
        line=dict(width=0.5, color='red'),
        stackgroup='one'
    ), row=1, col=1)
    stackfig.add_trace(go.Scatter(
        x=year, y=nyt_neu_val,
        name='neu score',
        mode='lines',
        line=dict(width=0.5, color='blue'),
        stackgroup='one'
    ), row=1, col=1)

    # add stacked scatters for nyp vals in row 1, col 2
    # set all layout parameters
    stackfig.add_trace(go.Scatter(
        x=year, y=nyp_pos_val,
        name='pos score',
        mode='lines',
        line=dict(width=0.5, color='green'),
        stackgroup='one',
        groupnorm='percent',  # sets the normalization for the sum of the stackgroup
        showlegend=False
    ), row=1, col=2)
    stackfig.add_trace(go.Scatter(
        x=year, y=nyp_neg_val,
        name='neg score',
        mode='lines',
        line=dict(width=0.5, color='red'),
        stackgroup='one',
        showlegend=False
    ), row=1, col=2)
    stackfig.add_trace(go.Scatter(
        x=year, y=nyp_neu_val,
        name='neu score',
        mode='lines',
        line=dict(width=0.5, color='blue'),
        stackgroup='one',
        showlegend=False
    ), row=1, col=2)

    # update layout
    stackfig.update_layout(
        showlegend=True,
        title_text="Vader Sentiment Stack Plot",
        xaxis_type='category',
        yaxis=dict(
            type='linear',
            range=[1, 100],
            ticksuffix='%'),
        xaxis2_type='category',
        yaxis2=dict(
            type='linear',
            range=[1, 100],
            ticksuffix='%'))

    # adds a vertical line delineating an important event relating to the current topic
    for k, v in topic_dict.items():
        if topic == k:
            for i in range(len(v)):
                stackfig.add_vline(x=v[i]['x_vline'], line_width=3, line_dash="dash", line_color="black", layer='above')
                stackfig.add_annotation(x=v[i]['x_annotation'], text=v[i]['text'], showarrow=False, textangle=-90)
                stackfig.add_annotation(x=v[i]['x_annotation'], xref='x2', text=v[i]['text'], showarrow=False, textangle=-90)
    # return fig
    return stackfig


def word_normalization(nyt_texts, nyp_texts):
    """

    Parameters
    ----------
    nyt_texts
    nyp_texts

    Returns
    -------

    """
    stops = list(set(stopwords.words('english')))
    df = pd.DataFrame(columns=["text", "news_outlet"])

    for year in range(2002, 2023):
        nyt_string = nyt_texts[year]
        nyp_string = nyp_texts[year]

        # remove numbers from strings
        nyt_string = re.sub(r'\d+', '', nyt_string)
        nyp_string = re.sub(r'\d+', '', nyp_string)

        # remove other punctuation
        nyt_string = re.sub(r'[^\w\s]', '', nyt_string)
        nyp_string = re.sub(r'[^\w\s]', '', nyp_string)

        # split strings
        nyt = nyt_string.split()
        nyp = nyp_string.split()

        # remove stop words
        for word in stops:
            nyt = [value for value in nyt if value != word]
            nyp = [value for value in nyp if value != word]

        # rejoin to string
        nyt = ' '.join(nyt)
        nyp = ' '.join(nyp)

        df = df.append({"text": nyt, "news_outlet": "nyt"}, ignore_index=True)
        df = df.append({"text": nyp, "news_outlet": "nyp"}, ignore_index=True)

    return df


def topic_modelling(df):
    # initialize count vectorizer
    cv = CountVectorizer(max_df=.95, min_df=2)

    # create document term matrix
    dtm = cv.fit_transform(df["text"])

    # build lda model and fit to document term matrix
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)

    total_str = ''
    # retrieve words of each topic
    for i, topic in enumerate(lda.components_):
        total_str = total_str + f"The top 15 words for topic #{i}\n"
        words = [cv.get_feature_names_out()[index] for index in topic.argsort()[-15:]]
        topics_str = ', '.join(str(item) for item in words)
        total_str = total_str + topics_str + '\n\n'

    topic_results = lda.transform(dtm)
    df["topic"] = topic_results.argmax(axis=1)

    return df, total_str


def topic_modelling_graph(df, topic_title):
    topics = df["topic"].unique()
    topics.sort()

    nyt_dict = {}
    nyp_dict = {}
    for topic in topics:
        nyt_dict[topic] = 0
        nyp_dict[topic] = 0

    nyt_df = df[df["news_outlet"] == "nyt"]
    nyt_counts = nyt_df.groupby(["topic"]).size().reset_index(name='counts')

    for index, row in nyt_counts.iterrows():
        if row["topic"] in nyt_dict:
            nyt_dict[row["topic"]] = row["counts"]

    nyp_df = df[df["news_outlet"] == "nyp"]
    nyp_counts = nyp_df.groupby(["topic"]).size().reset_index(name='counts')

    for index, row in nyp_counts.iterrows():
        if row["topic"] in nyp_dict:
            nyp_dict[row["topic"]] = row["counts"]

    nyt_vals = list(nyt_dict.values())
    nyp_vals = list(nyp_dict.values())

    fig = go.Figure(data=[
        go.Bar(name="NYT", x=topics, y=nyt_vals),
        go.Bar(name="NYP", x=topics, y=nyp_vals)
    ])
    title = f"NYT vs NYP Topic Modelling Comparison: {topic_title}"
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.update_layout(title=title,
                      xaxis=dict(title="Topic Number"),
                      yaxis=dict(title="Frequency")
                      )
    return fig


def main():
    # establish title and topic
    topic = 'abortion'
    title = 'NLP Comparison of Prevalent Social Issues: ' + topic.capitalize()
    topic_graph_title = 'Sub Topics Generated Through Topic Modelling for ' + topic.capitalize()

    nyt_imagefile = 'NYTLogo.jpeg'
    nyt_encoded = base64.b64encode(open(nyt_imagefile, 'rb').read()).decode('ascii')

    nyp_imagefile = 'NYPLogo.jpeg'
    nyp_encoded = base64.b64encode(open(nyp_imagefile, 'rb').read()).decode('ascii')

    # load texts for topic
    nyt_texts, nyp_texts, df = load_topic(2002, 2022, topic, negs)

    # topic modelling functions
    words_df = word_normalization(nyt_texts, nyp_texts)
    words_df, text_str = topic_modelling(words_df)

    # make vis figures
    s = sank(8, 2013, topic, negs)
    p = parallel(2002, 2002, topic, negs)
    w1, w2 = wordcloud(nyt_texts, nyp_texts)
    st = stacked(df, topic, topics, negs)
    bar = topic_modelling_graph(words_df, topic)

    # range of years, making the steps for the marks
    min_year = 2002
    max_year = 2022
    poss_years = list(range(min_year, max_year))
    marks = {}
    for num in range(min_year, max_year + 1, 1):
        marks[num] = str(num)

    # make app
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = Dash(__name__, external_stylesheets=external_stylesheets)
    app.layout = html.Div(id='App', children=[

        # top line of Dash
        html.Div([

            # first logo
            html.Div([
                html.Img(src='data:image/png;base64,{}'.format(nyt_encoded),
                         style={'height': '80%', 'width': '80%'})],
                style={'width': '25%', 'display': 'inline'}),

            # set title
            html.Div(id='title_words', children=[
                html.H1(title, style={'text-align': 'center'})],
                     style={'width': '50%', 'display': 'inline'}),

            # second logo
            html.Div([
                html.Img(src='data:image/png;base64,{}'.format(nyp_encoded),
                         style={'height': '80%', 'width': '100%'})],
                style={'width': '25%', 'display': 'inline'})],
            style={'display': 'flex'}),

        # dropdown for topics
        html.Div(id='Topic', children=[
            dcc.Dropdown(id='topic_selector',
                         options=['abortion', 'gay marriage', 'marijuana'],
                         value='abortion',
                         style={'width': '40%'})],
                 style={'align-items': 'center', 'justify-content': 'center'}),

        dcc.Tabs([
            # sankey figure with dropdown
            dcc.Tab(label='Sankey Diagram by Year', children=[
                html.H4('Most Common Words', style={'text-align': 'center'}),
                dcc.Graph(id='sankey_fig', figure=s),
                dcc.Dropdown(id='sankey_year', options=poss_years, value=2013)],
                    style={'backgroundColor': '#207947'}),

            # parallel fig with slider
            dcc.Tab(label='Parallel Coordinate Sentiment Comparison', children=[
                html.H4('Sentiment Parallel Coordinates', style={'text-align': 'center'}),
                dcc.Graph(id='parallel_fig', figure=p),
                dcc.Slider(id='parallel_year', min=min_year, max=max_year, step=None, marks=marks, value=2002)],
                    style={'backgroundColor': '#207947'}),

            # wordcloud figs
            dcc.Tab(label='Wordcloud Comparison', children=[
                html.Div(id='Wordclouds', children=[
                    html.Div([dcc.Graph(id='wordcloud_nyt', figure=w1)], style={'width': '50%', 'display': 'inline'}),
                    html.Div([dcc.Graph(id='wordcloud_nyp', figure=w2)],
                             style={'width': '50%', 'display': 'inline'}), ],
                         style={'display': 'flex'})], style={'backgroundColor': '#207947'}),

            # stacked fig
            dcc.Tab(label='Stacked Plot', children=[
                dcc.Graph(id='stacked', figure=st)], style={'backgroundColor': '#207947'}),

            # topic modelling graph
            dcc.Tab(label='Topic Modelling Comparison', children=[
                html.H4(topic_graph_title, style={'text-align': 'center'}),
                dcc.Textarea(id='sub_topics', value=text_str,
                             style={'text-align': 'center', 'align-items': 'center', 'width': '100%', 'height': 300}),
                dcc.Graph(id='barchart', figure=bar)],
                    style={'backgroundColor': 'rgb(161,212,173)'})
        ])],
                          style={'backgroundColor': '#207947'})


# callbacks
    @app.callback(
        Output(component_id='title_words', component_property='children'),
        Input(component_id='topic_selector', component_property='value')
    )
    def update_header(topic_selector):
        # make title and header
        title = 'NLP Comparison of Prevalent Social Issues: ' + topic_selector.capitalize()
        header = html.H1(title, style={'text-align': 'center'})
        return header

    @app.callback(
        Output(component_id='sankey_fig', component_property='figure'),
        Input(component_id='sankey_year', component_property='value'),
        Input(component_id='topic_selector', component_property='value')
    )
    def update_sankey(sankey_year, topic_selector):
        # updates sankey based on new topic and year
        s = sank(8, sankey_year, topic_selector, negs)
        return s

    @app.callback(
        Output(component_id='parallel_fig', component_property='figure'),
        Input(component_id='parallel_year', component_property='value'),
        Input(component_id='topic_selector', component_property='value')
    )
    def update_parallel(parallel_year, topic_selector):
        p = parallel(parallel_year, parallel_year, topic_selector, negs)
        return p

    @app.callback(
        Output(component_id='wordcloud_nyt', component_property='figure'),
        Input(component_id='topic_selector', component_property='value')
    )
    def update_wordcloud_nyt(topic_selector):
        #nyt_texts, nyp_texts, df = load_topic(min_year, max_year, topic_selector, negs)
        w1, w2 = wordcloud(nyt_texts, nyp_texts)
        return w1

    @app.callback(
        Output(component_id='wordcloud_nyp', component_property='figure'),
        Input(component_id='topic_selector', component_property='value')
    )
    def update_wordcloud_nyp(topic_selector):
        #nyt_texts, nyp_texts, df = load_topic(min_year, max_year, topic_selector, negs)
        w1, w2 = wordcloud(nyt_texts, nyp_texts)
        return w2

    @app.callback(
        Output(component_id='stacked', component_property='figure'),
        Input(component_id='topic_selector', component_property='value')
    )
    def update_stacked(topic_selector):
        st = stacked(df, topic_selector, topics, negs)
        return st

    @app.callback(
        Output(component_id='barchart', component_property='figure'),
        Input(component_id='topic_selector', component_property='value')
    )
    def update_bar(topic_selector):
        #
        # for time in range(0, 1):
        #     nyt_texts, nyp_texts, df = load_topic(min_year, max_year, topic_selector, negs)

        # topic modelling functions
        words_df = word_normalization(nyt_texts, nyp_texts)
        words_df, text_str = topic_modelling(words_df)

        # update bar chart
        bar_graph = topic_modelling_graph(words_df, topic_selector)
        return bar_graph

    # run server
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
