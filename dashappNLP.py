from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import main
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
app = Dash(__name__)

negs = main.load_neg_words("negative-words.txt")
df = main.load_topic(2002, 2022, 'abortion', negs)

nyt_df = df[['year', 'nyt-pos', 'nyt-neg', 'nyt-neu', 'nyt-neg_ratio']]
nyt_df.rename(columns={'nyt-pos':'Positive Sentiment', 'nyt-neg':'Negative Sentiment', 'nyt-neu':'Neural Sentiment',
                       'nyt-neg_ratio':'Negative Sentiment Ratio'}, inplace=True)
nyt_df['Journal'] = [0]*len(nyt_df)

nyp_df = df[['year', 'nyp-pos', 'nyp-neg', 'nyp-neu', 'nyp-neg_ratio']]
nyp_df.rename(columns={'nyp-pos':'Positive Sentiment', 'nyp-neg':'Negative Sentiment', 'nyp-neu':'Neural Sentiment',
                       'nyp-neg_ratio':'Negative Sentiment Ratio'}, inplace=True)
nyp_df['Journal'] = [1]*len(nyp_df)

parallel_df = pd.concat([nyp_df, nyt_df]).sort_values('year')
parallel_fig = px.parallel_coordinates(parallel_df, color='Journal',
                                       dimensions=['Positive Sentiment', 'Negative Sentiment', 'Neural Sentiment',
                                                   'Negative Sentiment Ratio'],
                                       color_continuous_scale=px.colors.diverging.Tealrose,
                                       color_continuous_midpoint=0.5
                                       )

app.layout = html.Div([
    dcc.Graph(id='parallel_coordinates', figure=parallel_fig)
])

app.run_server(debug=True)