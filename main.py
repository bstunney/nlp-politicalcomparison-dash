from political_nlp import pnlp
import pandas as pd
import plotly.graph_objects as go
import itertools
import sankey
import matplotlib.pyplot as plt

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
from plotly.subplots import make_subplots

import re as re
import string
import wordcloud as wc
import nltk
from nltk.corpus import stopwords
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
    for i in range(start, end+1):

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
        for k,v in nyt.text.items():
            new = re.sub('[' + string.punctuation + ']', '', v).lower().strip()
            nyt_str += new

        # add nyt_str to nyt_passages dct
        nyt_passages[i] = nyt_str

        # make pnlp object for nyt year of interest and load text
        nyp= pnlp(f"{topic}-nyp/nyp_{topic}_{i}.json", negs)
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
                 "nyp-neg": [nyp.neg], "nyp-neu": [nyp.neu], "nyt-neg_ratio":[nyt.neg_ratio],
                 "nyp-neg_ratio":[nyp.neg_ratio]})

        # otherwise, make a new df for each year
        else:
            ndf = pd.DataFrame(
                {'year': [i], "nyt-pos": [nyt.pos], "nyt-neg": [nyt.neg], "nyt-neu": [nyt.neu], "nyp-pos": [nyp.pos],
                 "nyp-neg": [nyp.neg], "nyp-neu": [nyp.neu], "nyt-neg_ratio":[nyt.neg_ratio],
                 "nyp-neg_ratio":[nyp.neg_ratio]})

            # concat new df to first year of interest df
            df = pd.concat([df,ndf])

        # no longer = first iteration
        first = False

    # return nyt, nyp passages and stats df
    return nyt_passages, nyp_passages, df

def wordcloud(nyt_texts, nyp_texts, year):
    """

    :param nyt_texts:
    :param nyp_texts:
    :param year:
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
    :return: parellel_fig: figure
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

def stacked(df, topic, negs):
    """
    make stacked scatter vis fig
    :param df: (dataframe) df of all sentiment score averages for each year of a topic byt nyt and nyp
    :param topic: (str) topic of interest
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

    # return fig
    return stackfig

def main():

    # establish title and topic
    topic = 'abortion'
    title = f'NLP Comparison of Prevalent Social Issues: {" ".join([x.capitalize() for x in topic.split()])}'

    nyt_imagefile = 'NYTLogo.jpeg'
    nyt_encoded = base64.b64encode(open(nyt_imagefile, 'rb').read()).decode('ascii')

    nyp_imagefile = 'NYPLogo.jpeg'
    nyp_encoded = base64.b64encode(open(nyp_imagefile, 'rb').read()).decode('ascii')

    # load texts for topic
    nyt_texts, nyp_texts, df = load_topic(2002, 2022, topic, negs)

    # make vis figures
    s = sank(8, 2013, topic, negs)
    p = parallel(2002, 2002, topic, negs)
    w1, w2 = wordcloud(nyt_texts, nyp_texts, topic)
    st = stacked(df, topic, negs)

    # range of years
    poss_years = list(range(2002, 2023))

    # make app
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = Dash(__name__, external_stylesheets=external_stylesheets)
    app.layout = html.Div(id='App', children=[

        # top line of Dash
        html.Div([

            # first logo
            html.Div([
                html.Img(src='data:image/png;base64,{}'.format(nyt_encoded),
                         style={'height':'80%', 'width':'80%'})],
                style={'width': '25%', 'display': 'inline'}),

            # set title
            html.Div(id='title_words', children=[
                html.H1(title, style={'text-align': 'center'})],
                     style={'width':'50%', 'display':'inline'}),

            # second logo
            html.Div([
                html.Img(src='data:image/png;base64,{}'.format(nyp_encoded),
                         style={'height':'80%', 'width':'100%'})],
                style={'width': '25%', 'display': 'inline'})],
        style={'display':'flex'}),

        # dropdown for topics
        html.Div(id='Topic', children=[
            dcc.Dropdown(id='topic-selector',
                         options=['abortion', 'gay marriage', 'marijuana'],
                         value='abortion',
                         style={'width':'40%'})],
                 style={'align-items': 'center', 'justify-content': 'center'}),

        dcc.Tabs([
        # sankey figure with dropdown
        dcc.Tab(label='Sankey Diagram by Year', children=[
            html.H4('Most Common Words', style={'text-align': 'center'}),
            dcc.Graph(id='sankey_fig', figure=s),
            dcc.Dropdown(id='sankey_year', options=poss_years, value=2013)],
                style={'backgroundColor':'#207947'}),

        # parallel fig with slider
        dcc.Tab(label='Parallel Coordinate Sentiment Comparison', children=[
            html.H4('Sentiment Parallel Coordinates', style={'text-align': 'center'}),
            dcc.Graph(id='parallel_fig', figure=p),
            dcc.Slider(id='parallel_year', min=2002, max=2022, step=1, value=2002,
                            marks={opacity: f'{opacity:.0f}' for opacity in poss_years})]),

        # wordcloud figs
        dcc.Tab(label='Wordcloud Comparison', children=[
            html.Div(id='Wordclouds', children=[
            html.Div([dcc.Graph(id='wordcloud_nyt', figure=w1)], style={'width': '50%', 'display': 'inline'}),
            html.Div([dcc.Graph(id='wordcloud_nyp', figure=w2)], style={'width': '50%', 'display': 'inline'})],
                 style={'display': 'flex'})]),

        # stacked fig
        dcc.Tab(label='Stacked Plot', children=[
            dcc.Graph(id='stacked', figure=st)]),
    ])],
                          style={'backgroundColor':'#207947'})

    # callback
    @app.callback(
        Output(component_id='sankey_fig', component_property='figure'),
        Output(component_id='parallel_fig', component_property='figure'),
        Output(component_id='wordcloud_nyt', component_property='figure'),
        Output(component_id='wordcloud_nyp', component_property='figure'),
        Output(component_id='stacked', component_property='figure'),
        Output(component_id='title_words', component_property='children'),
        [Input(component_id='sankey_year', component_property='value'),
         Input(component_id='parallel_year', component_property='value'),
         Input(component_id='topic-selector', component_property='value')]
    )
    def _refresh_visualizations(sankey_year, parallel_year, topic):
        """
        reload texts for new topix
        :param sankey_year: (int) year of interest
        :param parallel_years:
        :param topic: (str) topic of interest
        :return: all figs
        """

        # reload data
        nyt_texts, nyp_texts, df = load_topic(2002, 2022, topic, negs)

        # make figs
        s = sank(8, sankey_year, topic, negs)
        p = parallel(parallel_year, parallel_year, topic, negs)
        w1, w2 = wordcloud(nyt_texts, nyp_texts, topic)
        st = stacked(df, topic, negs)

        # make title and header
        title = f'NLP Comparison of Prevalent Social Issues: {" ".join([x.capitalize() for x in topic.split()])}'
        header = html.H1(title, style={'text-align': 'center'})

        return s, p, w1, w2, st, header

    # run server
    app.run_server(debug=True)

if __name__ == "__main__":
    main()