from political_nlp import pnlp
import pandas as pd
import plotly.graph_objects as go



def load_neg_words(filename):
    f = open(filename, "r")
    lst = f.readlines()
    for i in range(len(lst)):
        lst[i] = lst[i].replace("\n", "")

    lst = lst[35:]

    return lst

def load_topic(start, end, topic, negs):

    first = True
    for i in range(start, end+1):

        print("Loading for year:", i)
        lst = []
        lst.append(i)

        nyt = pnlp(f"{topic}-nyt/nyt_{topic}_{i}.json", negs)
        nyt.load_text("nyt")

        nyp= pnlp(f"{topic}-nyp/nyp_{topic}_{i}.json", negs)
        nyp.load_text("nyp")

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

    return df

def bars(colors, x_data, y_data):

    # https://plotly.com/python/horizontal-bar-charts/

    fig = go.Figure()

    for i in range(0, len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            fig.add_trace(go.Bar(
                x=[xd[i]], y=[yd],
                orientation='h',
                marker=dict(
                    color=colors[i],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            ))

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=[0.15, 1]
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        barmode='stack',
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        margin=dict(l=120, r=10, t=140, b=80),
        showlegend=False,
    )

    fig.show()

def main():



    """
    lsts = df.values.tolist()
    colors = ["red", "green", "yellow"]

    y_data = []
    y_data_r = []
    for i in range(len(lsts)):
        y_data.append(lsts[i][0])
        lsts[i] = lsts[i][1:4]
        l = lsts[i][4:]
        y_data_r.append(l)
    x_data = lsts

    print(x_data)
    print(y_data)

    bars(colors, x_data, y_data)




    bars(colors, x_data, y_data_r)
    """

    negs = load_neg_words("negative-words.txt")
    df = load_topic(2002, 2022, "abortion", negs)
    print(df)



    #nyt_a_2013 = pnlp("abortion-nyt/nyt_abortion_2013.json")
    #nyt_a_2013.load_text("nyt")

    #nyp_a_2013 = pnlp("abortion-nyp/nyp_abortion_2013.json")
    #nyp_a_2013.load_text("nyp")

if __name__ == "__main__":
    main()