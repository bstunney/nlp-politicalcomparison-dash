from political_nlp import pnlp
import pandas as pd
import plotly.graph_objects as go
import itertools
import sankey
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from dash import Dash, html, dcc, Input, Output
import plotly.express as px

import re as re
import string
import wordcloud as wc
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore")
app = Dash(__name__)

def load_neg_words(filename):
    f = open(filename, "r")
    lst = f.readlines()
    for i in range(len(lst)):
        lst[i] = lst[i].replace("\n", "")

    lst = lst[35:]

    return lst
negs = load_neg_words("negative-words.txt")
def load_topic(start, end, topic, negs):

    nyt_passages = {}
    nyp_passages = {}
    first = True
    for i in range(start, end+1):

        print("Loading for year:", i)
        lst = []
        lst.append(i)

        nyt = pnlp(f"{topic}-nyt/nyt_{topic}_{i}.json", negs)
        nyt.load_text("nyt")
        nyt_str = ""
        for k,v in nyt.text.items():
            new = re.sub('[' + string.punctuation + ']', '', v).lower().strip()
            nyt_str += new
        nyt_passages[i] = nyt_str

        nyp= pnlp(f"{topic}-nyp/nyp_{topic}_{i}.json", negs)
        nyp.load_text("nyp")
        nyp_str = ""
        for k, v in nyp.text.items():
            new = re.sub('[' + string.punctuation + ']', '', v).lower().strip()
            nyp_str += new
        nyp_passages[i] = nyp_str

        if first == True:
            df = pd.DataFrame(
                {'year': [i], "nyt-pos": [nyt.pos], "nyt-neg": [nyt.neg], "nyt-neu": [nyt.neu], "nyp-pos": [nyp.pos],
                 "nyp-neg": [nyp.neg], "nyp-neu": [nyp.neu], "nyt-neg_ratio":[nyt.neg_ratio],
                 "nyp-neg_ratio":[nyp.neg_ratio]})
        else:
            ndf = pd.DataFrame(
                {'year': [i], "nyt-pos": [nyt.pos], "nyt-neg": [nyt.neg], "nyt-neu": [nyt.neu], "nyp-pos": [nyp.pos],
                 "nyp-neg": [nyp.neg], "nyp-neu": [nyp.neu], "nyt-neg_ratio":[nyt.neg_ratio],
                 "nyp-neg_ratio":[nyp.neg_ratio]})

            df = pd.concat([df,ndf])
        first = False

    return nyt_passages, nyp_passages, df

def wordcloud(nyt_texts, nyp_texts, topic):

    # making list of stop words to remove
    stops = list(set(stopwords.words('english')))
    stops.append("s")

    nytstring = ''
    nypstring = ''

    for year in range(2002,2023):
        nytstring += nyt_texts[year]
        nypstring += nyp_texts[year]

    nyt = nytstring.split()
    nyp = nypstring.split()

    for word in stops:
        nyt = [value for value in nyt if value != word]
        nyp = [value for value in nyp if value != word]

    nyt = ' '.join(nyt)
    nyp = ' '.join(nyp)

    cloud = wc.WordCloud()

    nyt_cloud = cloud.generate(nyt)
    fig1 = go.Figure()
    fig1.add_trace(go.Image(z=nyt_cloud))
    fig1.update_layout(height=600, width=800, xaxis={'visible': False}, yaxis={'visible': False},
                       title_text='New York Times')

    nyp_cloud = cloud.generate(nyp)
    fig2 = go.Figure()
    fig2.add_trace(go.Image(z=nyp_cloud))
    fig2.update_layout(height=600, width=800, xaxis={'visible': False}, yaxis={'visible': False},
                             title_text='New York Post')

    return fig1, fig2

def parallel(start, end, topic, negs):
    nyt_texts, nyp_texts, df = load_topic(start, end, topic, negs)

    nyt_df = df[['year', 'nyt-pos', 'nyt-neg', 'nyt-neu', 'nyt-neg_ratio']]
    nyt_df.rename(
        columns={'nyt-pos': 'Positive Sentiment', 'nyt-neg': 'Negative Sentiment', 'nyt-neu': 'Neutral Sentiment',
                 'nyt-neg_ratio': 'Negative Sentiment Ratio'}, inplace=True)
    nyt_df['Journal'] = [0] * len(nyt_df)

    nyp_df = df[['year', 'nyp-pos', 'nyp-neg', 'nyp-neu', 'nyp-neg_ratio']]
    nyp_df.rename(
        columns={'nyp-pos': 'Positive Sentiment', 'nyp-neg': 'Negative Sentiment', 'nyp-neu': 'Neutral Sentiment',
                 'nyp-neg_ratio': 'Negative Sentiment Ratio'}, inplace=True)
    nyp_df['Journal'] = [1] * len(nyp_df)

    parallel_df = pd.concat([nyp_df, nyt_df]).sort_values('year')
    parallel_fig = px.parallel_coordinates(parallel_df, color='Journal',
                                           dimensions=['Positive Sentiment', 'Negative Sentiment', 'Neutral Sentiment',
                                                       'Negative Sentiment Ratio'],
                                           color_continuous_scale=px.colors.diverging.Tealrose,
                                           color_continuous_midpoint=0.5
                                           )

    return parallel_fig

def sank(num_words, year, topic, negs):

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
    cols = list(df.index)

    year = []

    nyt_pos_val = []
    nyt_neg_val = []
    nyt_neu_val = []

    nyp_pos_val = []
    nyp_neg_val = []
    nyp_neu_val = []
    for i in range(len(df)):
        year.append(df.iloc[i]['year'])
        nyt_pos_val.append(df.iloc[i]['nyt-pos'])
        nyp_pos_val.append(df.iloc[i]['nyp-pos'])

        nyt_neg_val.append(df.iloc[i]['nyt-neg'])
        nyp_neg_val.append(df.iloc[i]['nyp-neg'])

        nyt_neu_val.append(df.iloc[i]['nyt-neu'])
        nyp_neu_val.append(df.iloc[i]['nyp-neu'])

    stackfig = make_subplots(rows=1, cols=2,
                        subplot_titles=("NY Times", "NY Post"))

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

    return stackfig


def main():
    topic = 'abortion'
    title = f'NLP Comparison of Prevalent Social Issues: {topic}'

    nyt_texts, nyp_texts, df = load_topic(2002, 2022, topic, negs)

    nyp_a_2013 = pnlp("abortion-nyp/nyp_abortion_2013.json", negs)
    nyp_a_2013.load_text("nyp")
    dct1 = nyp_a_2013.noun_scores

    nyt_a_2013 = pnlp("abortion-nyt/nyt_abortion_2013.json", negs)
    nyt_a_2013.load_text("nyt")
    dct2 = nyt_a_2013.noun_scores


    s = sank(8, 2013, topic, negs)
    p = parallel(2002, 2022, topic, negs)
    w1, w2 = wordcloud(nyt_texts, nyp_texts, topic)
    st = stacked(df, topic, negs)

    poss_years = list(range(2002,2023))

    app = Dash(__name__)
    app.layout = html.Div(id='App', children=[

        html.Div(id='title_words', children=[
            html.H1(title, style={'text-align': 'center'})]),

        html.Div(id='Topic', children=[
            dcc.Dropdown(id='topic-selector',
                         options=['abortion', 'gay marriage', 'marijuana'],
                         value='abortion')]),

        html.Div(id='Sankey', children=[
            html.H4('Most Common Words', style={'text-align': 'center'}),
            dcc.Graph(id='sankey_fig', figure=s),
            dcc.Dropdown(id='sankey_year', options=poss_years, value=2013)]),

        html.Div(id='Parallel', children=[
            html.H4('Sentiment Parallel Coordinates', style={'text-align': 'center'}),
            dcc.Graph(id='parallel_fig', figure=p),
            dcc.RangeSlider(id='parallel_years', min=2002, max=2022, step=1, value=[2002, 2022],
                            marks={opacity: f'{opacity:.0f}' for opacity in poss_years})]),

        html.Div(id='Wordcloud', children=[
            html.Div([dcc.Graph(id='wordcloud_nyt', figure=w1)], style={'width': '50%', 'display': 'inline'}),
            html.Div([dcc.Graph(id='wordcloud_nyp', figure=w2)], style={'width': '50%', 'display': 'inline'})],
                    style={'display': 'flex'}),

        html.Div(id='Stacked', children=[
            dcc.Graph(id='stacked', figure=st)]),
    ])

    @app.callback(
        Output(component_id='sankey_fig', component_property='figure'),
        Output(component_id='parallel_fig', component_property='figure'),
        Output(component_id='wordcloud_nyt', component_property='figure'),
        Output(component_id='wordcloud_nyp', component_property='figure'),
        Output(component_id='stacked', component_property='figure'),
        Output(component_id='title_words', component_property='children'),
        [Input(component_id='sankey_year', component_property='value'),
         Input(component_id='parallel_years', component_property='value'),
         Input(component_id='topic-selector', component_property='value')]
    )

    def _refresh_visualizations(sankey_year, parallel_years, topic):
        nyt_texts, nyp_texts, df = load_topic(2002, 2022, topic, negs)

        s = sank(8, sankey_year, topic, negs)
        p = parallel(parallel_years[0], parallel_years[1], topic, negs)
        w1, w2 = wordcloud(nyt_texts, nyp_texts, topic)
        st = stacked(df, topic, negs)
        title = f'NLP Comparison of Prevalent Social Issues: {topic}'
        header = html.H1(title, style={'text-align': 'center'})

        return s, p, w1, w2, st, header


    app.run_server(debug=True)

if __name__ == "__main__":
    main()