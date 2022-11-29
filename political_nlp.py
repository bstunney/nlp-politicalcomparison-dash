from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from textblob import TextBlob
import nltk
import itertools
#nltk.download('brown')
#nltk.download('punkt')


class pnlp:

    def __init__(self, filename, negatives):

        # take in file name
        # construct pnlp vals
        self.data = filename

        self.neg_words = negatives
        self.neg_ratio = 0
        self.nouns = []

        self.pos = 0
        self.neg = 0
        self.neu = 0
        self.text = ''

        self.word_length = 0
        self.noun_scores = {}

    @staticmethod
    def _nyt_parser(self):
        """ parser for dirty nyt json files

        :param self: self
        :return: dct (dict): dct with cleaned articles
        """

        # open file and use json.load
        f = open(self.data)
        raw = json.load(f)

        # store all k/v pairs where text length is 0
        del_lst = []
        for k, v in raw.items():
            if len(v) == 0:
                del_lst.append(k)

        # delete all fields of 0
        for i in range(len(del_lst)):
            del raw[del_lst[i]]

        # iterate through each of the articles
        for k, v in raw.items():

            # get article
            article = v

            # start after lyrics header
            article = article.replace("\\", "")

            raw[k] = article

        return raw

    @staticmethod
    def _nyp_parser(self):
        """ parser for dirty nyp json files

        :param self: self
        :return: dct (dict): dct with cleaned articles
        """

        # open file and use json.load
        f = open(self.data)
        raw = json.load(f)

        # store all k/v pairs where text length is 0
        del_lst = []
        for k, v in raw.items():
            if len(v) == 0:
                del_lst.append(k)

        # delete all fields of 0
        for i in range(len(del_lst)):
            del raw[del_lst[i]]

        # iterate through each of the articles
        for k, v in raw.items():

            # get article
            article = v

            # start after lyrics header
            article = article.replace("<strong>", "").replace("</strong>", "").replace("<em>", "").replace("</em>", "")
            article = article.replace("</a>", "")
            article = article.replace("\n","").replace("*","")

            # if article has href section, clean
            if "<a href=" in article:
                s = article.split("<a href=")

                # add all cleaned strings
                str = ''
                for i in range(1, len(s)):
                    after = s[i].split(">")
                    str += " ".join(after[1:])

                # set article to removed href version
                article = str

            # redefine k,v pair
            raw[k] = article

        # return dict
        return raw

    def load_text(self, parser="nyt"):
        """ registers the text file with the NLP framework
        :param parser (string):  specifies if file is nyp or nyt
        :return: none
        """

        # default parser
        if parser == "nyp":
            dct = pnlp._nyp_parser(self)

        # json parser
        elif parser == "nyt":
            dct = pnlp._nyt_parser(self)

        # call save results depending on list of strings of words from parsers
        self.text = dct
        self._save_results(dct)

    def _save_results(self, dct):
        """ saves sentiment scores for year/news source

        :param results (lst): list of strings of phrases or full song lyrics
        :return: none
        """

        pos = []
        neg = []
        neu = []
        neg_rat = []
        noun_lst = []
        lengths = []

        for k, v in dct.items():
            # Create a SentimentIntensityAnalyzer object.
            sid_obj = SentimentIntensityAnalyzer()

            # polarity_scores method of SentimentIntensityAnalyzer
            # object returns sentiment dict with positivity, negativity, and neutrality
            sentiment_dict = sid_obj.polarity_scores(v)

            pos.append(sentiment_dict['pos'])
            neg.append(sentiment_dict['neg'])
            neu.append(sentiment_dict['neu'])

            count = 0
            for word in v.split(" "):
                if word in self.neg_words:
                    count += 1

            neg_rat.append(count/len(v.split(" ")))

            lengths.append(len(v.split(" ")))

            blob = TextBlob(v)
            noun_lst.append(list(blob.noun_phrases))


        self.word_length = sum(lengths)
        self.nouns = noun_lst
        self.neg_ratio = sum(neg_rat)/len(neg_rat)
        # return person's positivity score (neu for neutral if interested)
        self.pos = sum(pos) / len(pos)
        self.neg = sum(neg) / len(neg)
        self.neu = sum(neu) / len(neu)

        nouns = list(itertools.chain.from_iterable(noun_lst))

        noun_dct = {}
        for word in nouns:
            if "<" not in word and ">" not in word and word != '’ s' and '"' not in word:
                if word not in noun_dct:
                    noun_dct[word] = 1
                else:
                    noun_dct[word] += 1

        """
        for k,v in dct.items():
            dct[k] = v / nyt_a_2013.word_length
        """

        self.noun_scores = dict(sorted(noun_dct.items(), key=lambda item: item[1], reverse=True))
