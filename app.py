#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://github.com/mbr/flask-bootstrap.git
form to Database
Vanaf 70 naar multi
"""
from __future__ import division
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from flask import Flask, send_from_directory
from multiprocessing import Process
from goose3 import Goose
from textblob import TextBlob
from textatistic import Textatistic
import urllib.request
import re
import os
import time
import glob
import pandas as pd
import requests
from urllib.parse import urlsplit
from twitterscraper import query_tweets
from gensim.summarization import keywords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import pickle
import re
import numpy
from gensim import corpora, models
import gensim
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import os.path
import time
import json
from flask import Flask, render_template, request, redirect, session, flash, url_for, send_file, request, render_template
import os
import unicodedata
import pandas as pd
import time
from flask import send_file
import urllib
from MagicGoogle import MagicGoogle
from multiprocessing import Pool, cpu_count
from queue import Queue
from threading import Thread
import time
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from flask import Flask, send_from_directory
from multiprocessing import Process
import functools
from newspaper import Article
from gensim.summarization import keywords
from gensim.summarization import summarize
from gensim.summarization import keywords

import string
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


pd.set_option('precision', 2)
pd.options.display.max_colwidth = 999999999999999999999999999999999999

pickle_fname = 'pickle.model'
pickle_model = pickle.load (open (pickle_fname, 'rb'))

rec = re.compile(r"https?://(www\.)?")

def conv(s):
    try:
       return int(s)
    except ValueError:
       return s

def most_common(lst):
    return max(set(lst), key=lst.count)

app = Flask(__name__)
app.secret_key = 'ThisIsSecret'

def countLetters(word):
    count = 0
    for c in word:
        count += 1
    return count


def f1(q, url):
    try:
    #print("Start: %s" % time.ctime())
    # Instead of returning the result we put it in shared queue.
    #     st = "/&callback=process&key=57bf606e01a24537ac906a86dc27891f94a0f587"
    #     # zz = urlopen ( url )
    #     quez = 'http://api.mywot.com/0.4/public_link_json2?hosts=' + url + st
    #     stt = urllib.request.urlopen(quez).read()
    #     stt = str(stt)
    #     wot = re.findall('\d+', stt)
    #     ##z=[[conv(s) for s in line.split()] for line in wot]
    #     z = [conv(s) for s in wot]
    #     high = (z[1])
    #     low = (z[2])
    #     # print ( high , low )
        # WAYBACK
        zz = "{0.scheme}://{0.netloc}/".format(urlsplit(url))
        zurlz = "https://web.archive.org/web/0/" + str(zz)
        r = requests.get(zurlz, allow_redirects=False)
        data = r.content
        years = re.findall('\d+', str(data))
        years = [conv(s) for s in years]
        years = (years[0])
        years = int(str(years)[:4])
        cols = {'maturity': [years],
                # 'reputationLow': [low],
                # 'reputationHigh': [high],
                #'latency': [vals],
                'url': [str(url)]}
        dfb = pd.DataFrame.from_dict(cols)
        #print(dfb)
        #print("Start: %s" % time.ctime())
        q.put(dfb)
    except:
        #years=2018
        cols = {'maturity': [2018],
                # 'reputationLow': [low],
                # 'reputationHigh': [high],
                # 'latency': [vals],
                'url': [str(url)]}
        dfb = pd.DataFrame.from_dict(cols)
        # print(dfb)
        # print("Start: %s" % time.ctime())
        q.put(dfb)
    #     pass

def f2(q, url):
    try:
    #print("Start: %s" % time.ctime())
        #vals = requests.get(url, timeout=4, allow_redirects=False).elapsed.total_seconds()
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        #afb = len(article.images)
        blob = TextBlob(text)

        # taal = blob.detect_language()
        # if taal == ('en'):
        #     try:
        s = Textatistic(text)
        cols = {
                        'words': [s.word_count],
                        'pictures': [len(article.images)],
                        'subjectivity': [blob.sentiment.subjectivity],
                        'polarity': [blob.sentiment.polarity],
                        'readable': [s.flesch_score],
                        'text': [str(text)],
                        # 'kw': [ kw ] ,
                        'url': [str(url)]}
        dfa = pd.DataFrame.from_dict(cols)
                #print(dfa)
                #print("Start: %s" % time.ctime())
        q.put(dfa)
            # except:
    except:
        #s = Textatistic(text)
        cols = {
                        'words': [str('err')],
    #                    'latency': [str('err')],
                        'subjectivity': [str('err')],
                        'polarity': [str('err')],
                        'readable': [str('err')],
                        # 'kw': [ kw ] ,
                        'url': [str(url)]}
        dfa = pd.DataFrame.from_dict(cols)
                #print(dfa)
                #print("Start: %s" % time.ctime())
        q.put(dfa)
        #pass
            #     pass
    #pass


def f3(q, url):
    try:
    #print("Start: %s" % time.ctime())
    # Instead of returning the result we put it in shared queue.
        st = "/&callback=process&key=57bf606e01a24537ac906a86dc27891f94a0f587"
        # zz = urlopen ( url )
        quez = 'http://api.mywot.com/0.4/public_link_json2?hosts=' + url + st
        stt = urllib.request.urlopen(quez).read()
        stt = str(stt)
        wot = re.findall('\d+', stt)
        ##z=[[conv(s) for s in line.split()] for line in wot]
        z = [conv(s) for s in wot]
        high = (z[1])
        low = (z[2])
        # print ( high , low )
        # WAYBACK
        # zz = "{0.scheme}://{0.netloc}/".format(urlsplit(url))
        # zurlz = "https://web.archive.org/web/0/" + str(zz)
        # r = requests.get(zurlz, allow_redirects=False)
        # data = r.content
        # years = re.findall('\d+', str(data))
        # years = [conv(s) for s in years]
        # years = (years[0])
        # years = int(str(years)[:4])
        cols = {#'maturity': [years],
                'reputationLow': [low],
                'reputationHigh': [high],
                #'latency': [vals],
                'url': [str(url)]}
        dfc = pd.DataFrame.from_dict(cols)
        #print(dfb)
        #print("Start: %s" % time.ctime())
        q.put(dfc)
    except:
        #high = 0
        #low =0
        cols = {  # 'maturity': [years],
            'reputationLow': [0],
            'reputationHigh': [0],
            # 'latency': [vals],
            'url': [str(url)]}
        dfc = pd.DataFrame.from_dict(cols)
        # print(dfb)
        # print("Start: %s" % time.ctime())
        q.put(dfc)

def f4(q, url):
    try:
        vals = requests.get(url, timeout=4, allow_redirects=False).elapsed.total_seconds()
        # try:
        #print("Start: %s" % time.ctime())
        # Instead of returning the result we put it in shared queue.
        cols = {#'maturity': [years],
                    #'reputationLow': [low],
                    #'reputationHigh': [high],
                    'latency': [vals],
                    'url': [str(url)]}
        dfd = pd.DataFrame.from_dict(cols)
            #print(dfb)
            #print("Start: %s" % time.ctime())
        q.put(dfd)
    except:
        cols = {  # 'maturity': [years],
            # 'reputationLow': [low],
            # 'reputationHigh': [high],
            'latency': [str('err')],
            'url': [str(url)]}
        dfd = pd.DataFrame.from_dict(cols)
        pass

def f5(q, url):
    try:
            zzurl = rec.sub('', url).strip().strip('/')
            twtext = list ( )
            polar = list ( )
            datum = list ( )
            for tweet in query_tweets ( zzurl , 10 ):
                try:
                    txt = tweet.text
                    txt = re.sub ( r"http\S+" , "" , txt )
                    dat = tweet.timestamp
                    tblob = TextBlob ( txt )
                    tpol = tblob.sentiment.polarity
                    tal = tblob.detect_language()
                    if tal == ('en'):
                        twtext.append ( txt )
                        polar.append ( tpol )
                        datum.append ( dat )
                    else:
                        pass
                except:
                    pass


            df = pd.DataFrame ( {'tweet': twtext , 'timestamp': datum , 'polarity': polar} )
            df[ 'timestamp' ] = pd.to_datetime ( df[ 'timestamp' ] )
            oldest = df[ 'timestamp' ].min ( )
            newest = df[ 'timestamp' ].max ( )
            total = (oldest - newest).total_seconds ( )
            gem = total / len ( df.index )
            #df.to_csv ( 'sentiment.csv' , index=False , sep=',' , encoding='utf-8' )
            tmean = df[ "polarity" ].mean ( )
            tsd = df[ "polarity" ].std ( )
            tkur = df[ "polarity" ].kurtosis ( )
            ctweets = {'meansentiment': [ tmean ] ,
                    'sdpolarity': [ tsd ] ,
                    'kurtosispolarity': [ tkur ] ,
                    'tweetrate': [ gem ] ,
                    'tweetcount': [ len ( df.index ) ],
            'url': [str(url)] }
            dfe = pd.DataFrame.from_dict ( ctweets )

        #print(dfb)
        #print("Start: %s" % time.ctime())
            q.put(dfe)
    except:
        ctweets = {'meansentiment': ['err'],
                   'sdpolarity': [str('err')],
                   'kurtosispolarity': [str('err')],
                   'tweetrate': [str('err')],
                   'tweetcount': [str('err')],
            'url': [str(url)]}
        dfe = pd.DataFrame.from_dict(ctweets)
        q.put(dfe)

def tmpFunc(df):
    delayed_results = []
    for row in df.itertuples():
        #try:
            url=row.url
            result_queue = Queue()

            # One Thread for response time
            t1 = Thread(target=f1, args=(result_queue, url))
            #t5 = Thread (target=f5, args=(result_queue, url))
            t2 = Thread(target=f2, args=(result_queue, url))
            t3 = Thread(target=f3, args=(result_queue, url))
            t4 = Thread(target=f4, args=(result_queue, url))
            # Starting threads...
            #print("Start: %s" % time.ctime())

            t1.start()
            #t5.start()
            t2.start()
            t3.start()
            t4.start()

            # Waiting for threads to finish execution...
            t1.join(4)
            t2.join(4)
            t3.join(4)
            t4.join(4)
            #t5.join(4)
    #t.join()
            #print("End:   %s" % time.ctime())

            # After threads are done, we can read results from the queue.
            if not result_queue.empty():
                try:
                    r2 = result_queue.get(f2)
                    r1 = result_queue.get(f1)
                    r3 = result_queue.get(f3)
                    r4 = result_queue.get(f4)
                    #r5 = result_queue.get(f5)
                    #r4 = result_queue.get(f4)
                    #print('slot 1')
                        #print(r1)
                        #print(r1)
                    #print('Slot2')
                        #print(r2)
                    #mergen
                    #try:
                    #df=pd.merge(r1, r2, on='url')
                    dfs = [r1, r2, r3, r4]
                    df = functools.reduce(lambda left, right: pd.merge(left, right, on='url'), dfs)
                    #df = df[['mean', 4, 3, 2, 1]]
                    #print('dss')
                    #print(dss)
                    #print('struct')

                    #df = df[['readable', 'reputationHigh', 'reputationLow', 'polarity', 'latency', 'subjectivity', 'url', 'words', 'maturity']]
                    #df = df[['readable', 'reputationHigh', 'reputationLow', 'pictures', 'polarity', 'latency', 'subjectivity', 'url', 'words', 'maturity']]
                    #print (df)
                #return df
                except:
                    pass
                #print('df')
    return df



def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)
# def applyParallel(dfGrouped, func):
#     with Pool(cpu_count()) as p:
#         #retLst = p.map(func, [group for name, group in dfGrouped])
#         retLst = p.map(func, [group for name, group in dfGrouped])
#     return pd.concat(retLst)



@app.route('/getterm', methods=['POST', 'GET'])
def get_term():
    if request.method == 'POST':
        try:
            tag = request.form['srch-term']
            tt = tag

            mg = MagicGoogle()
            lijst = []
            #tt = 'Donald Trump'
            search = str(tt+' language:english file:html')
            for url in mg.search_url(query=search):
                lijst.append(url)

            df = pd.DataFrame({'url': lijst})
            #print('parallel versionOzzy: ')
            dff = ((applyParallel(df.groupby(df.index), tmpFunc)))
            dff = dff.query ('words != "err" & latency != "err"')
            #dff = dff.query ('words != "err" & latency != "err" & reputationHigh != "err" & maturity != "err"')
            #twit = dff[['kurtosispolarity', 'meansentiment', 'sdpolarity', 'tweetcount', 'tweetrate', 'url']]
            dff = dff[['readable', 'reputationHigh', 'reputationLow', 'pictures', 'polarity', 'latency', 'subjectivity',
                     'words', 'maturity', 'url']]
            newX = dff.values
            # newX=np.delete(newX, [1, 3], axis=1)
            newX = np.delete (newX, [9], axis=1)
            # print(newX)
            # newX = newX[~np.isnan(newX).any(axis=1)]
            # newX = newX.as_matrix().astype(np.float)
            # pickle_fname = 'pickle.model'
            # pickle_model = pickle.load (open (pickle_fname, 'rb'))
            result = pickle_model.predict (newX)  # print (result)
            px2 = result.reshape ((-1, 8))
            dffres = pd.DataFrame (
                {'complete': px2[:, 0], 'accuracy': px2[:, 1], 'precise': px2[:, 2], 'readable': px2[:, 3],
                 'relevant': px2[:, 4], 'trustworthy': px2[:, 5], 'overall': px2[:, 6], 'neutral': px2[:, 7]})
            return render_template('mp.html', dataframe=dff.to_html(index=False), res=dffres.to_html(index=False))
        except:
            flash ('We failed at finding matching URL Please try another Query', 'nameError')
            return redirect ('/')
tokenize = lambda doc: doc.lower().split(" ")
def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)

def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))

def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

def tfidf(documents):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

#in Scikit-Learn
#from sklearn.feature_extraction.text import TfidfVectorizer




def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

# @app.route('/getdetail', methods=['POST', 'GET'])
# def get_detail():
#     if request.method == 'POST':
#         try:
#             tag = request.form['srch-term']
#             tt = tag
#
#             mg = MagicGoogle()
#             lijst = []
#             #tt = 'Donald Trump'
#             search = str(tt+' language:english file:html')
#             for url in mg.search_url(query=search):
#                 lijst.append(url)
#
#             df = pd.DataFrame({'url': lijst})
#             #print('parallel versionOzzy: ')
#             dff = ((applyParallel(df.groupby(df.index), tmpFunc)))
#             dff = dff.query ('words != "err" & latency != "err"')
#             #dff.to_csv('tect.csv')
#             documents = dff['text'].tolist ()
#             #documents.to_csv('t')
#             dftekst= dff[['text', 'url']]
#             #dftekst.to_csv('ozzy.csv')
#             dftekst = dftekst.replace ('\n', ' ', regex=True)
#             #print(d)
#             ddurl = dff['url'].tolist ()
#             #dfurl = dff[['url']].copy()
#             #documents = dff['ur'].tolist ()
#             #print(documents)
#             vectorizer = TfidfVectorizer (stop_words='english')
#             X = vectorizer.fit_transform (documents)
#
#             true_k = 4
#             model = KMeans (n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
#             model.fit (X)
#
#             order_centroids = model.cluster_centers_.argsort ()[:, ::-1]
#             terms = vectorizer.get_feature_names ()
#
#             order_centroids = model.cluster_centers_.argsort ()[:, ::-1]
#             terms = vectorizer.get_feature_names ()
#             jk = []
#             for i in range (true_k):
#
#                 j = []
#                 jk.append (j)
#                 for ind in order_centroids[i, :10]:
#                     j.append (terms[ind])
#
#             cols = {'clusters': [jk]}
#             dfd = pd.DataFrame.from_dict (cols)
#             print(dfd)
#             #prediction = model.predict (X)
#             cols = {'Rowabove': model.predict (X),
#                     'urls': ddurl}
#             dfda = pd.DataFrame.from_dict (cols)
#
#             sklearn_tfidf = TfidfVectorizer (norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True,
#                                              tokenizer=tokenize)
#             sklearn_representation = sklearn_tfidf.fit_transform (documents)
#
#             tfidf_representation = tfidf (documents)
#             our_tfidf_comparisons = []
#             for count_0, doc_0 in enumerate (tfidf_representation):
#                 for count_1, doc_1 in enumerate (tfidf_representation):
#                     our_tfidf_comparisons.append ((cosine_similarity (doc_0, doc_1), count_0, count_1))
#
#             skl_tfidf_comparisons = []
#             for count_0, doc_0 in enumerate (sklearn_representation.toarray ()):
#                 for count_1, doc_1 in enumerate (sklearn_representation.toarray ()):
#                     skl_tfidf_comparisons.append ((cosine_similarity (doc_0, doc_1), count_0, count_1))
#
#             # print(sorted(skl_tfidf_comparisons, reverse = True))
#             score = (sorted (skl_tfidf_comparisons, reverse=True))
#             a = np.matrix (score)
#             #print(a)
#             dfsim = pd.DataFrame (a)
#             dfsim.columns = ['Similarity', 'docx', 'docy']
#             dfsim['docx'] = dfsim['docx'].astype(int)
#             dfsim['docy'] = dfsim['docy'].astype(int)
#
#             dff = dff[['readable', 'reputationHigh', 'reputationLow', 'pictures', 'polarity', 'latency', 'subjectivity',
#                      'words', 'maturity', 'url']]
#             newX = dff.values
#             newX = np.delete (newX, [9], axis=1)
#             result = pickle_model.predict (newX)
#             px2 = result.reshape ((-1, 8))
#             dffres = pd.DataFrame (
#                 {'complete': px2[:, 0], 'accuracy': px2[:, 1], 'precise': px2[:, 2], 'readable': px2[:, 3],
#                  'relevant': px2[:, 4], 'trustworthy': px2[:, 5], 'overall': px2[:, 6], 'neutral': px2[:, 7]})
#             #print('ozzy')
#             return render_template('detail.html', dataframe=dff.to_html(index=False), dkaey=dftekst.to_html(index=False), sim=dfsim.to_html(index=False), clres=dfda.to_html(index=False) ,clus=dfd.to_html(index=False), res=dffres.to_html(index=False))
#         except:
#             #print(e)
#             flash ('We failed at finding matching URL Please try another Query', 'nameError')
#             return redirect ('/')
#
#
# @app.route('/getdetail', methods=['POST', 'GET'])
# def get_detail():
#     if request.method == 'POST':
#         try:
#             tag = request.form['srch-term']
#             tt = tag
#
#             mg = MagicGoogle()
#             lijst = []
#             #tt = 'Donald Trump'
#             search = str(tt+' language:english file:html')
#             for url in mg.search_url(query=search):
#                 lijst.append(url)
#
#             df = pd.DataFrame({'url': lijst})
#             #print('parallel versionOzzy: ')
#             dff = ((applyParallel(df.groupby(df.index), tmpFunc)))
#             dff = dff.query ('words != "err" & latency != "err"')
#             #dff.to_csv('tect.csv')
#             documents = dff['text'].tolist ()
#             #documents.to_csv('t')
#             dftekst= dff[['text', 'url']]
#             #dftekst.to_csv('ozzy.csv')
#             dftekst = dftekst.replace ('\n', ' ', regex=True)
#             #print(d)
#             ddurl = dff['url'].tolist ()
#             #dfurl = dff[['url']].copy()
#             #documents = dff['ur'].tolist ()
#             #print(documents)
#             vectorizer = TfidfVectorizer (stop_words='english')
#             X = vectorizer.fit_transform (documents)
#
#             true_k = 4
#             model = KMeans (n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
#             model.fit (X)
#
#             # order_centroids = model.cluster_centers_.argsort ()[:, ::-1]
#             # terms = vectorizer.get_feature_names ()
#
#             order_centroids = model.cluster_centers_.argsort ()[:, ::-1]
#             terms = vectorizer.get_feature_names ()
#             jk = []
#             for i in range (true_k):
#
#                 j = []
#                 jk.append(j)
#                 for ind in order_centroids[i, :10]:
#                     j.append(terms[ind])
#
#
#
#             #cols = {'clusters': [jk]}
#             #jk=([' '.join (x) for x in jk])
#             #Out[4]: 'Hello\nWorld'
#             # #dfd = pd.DataFrame.from_dict (cols)
#             # '\n'.join ([''.join (x) for x in pic])
#             # Out[4]: 'Hello\nWorld'
#             dfd = pd.DataFrame.from_dict (jk)
#             #dfd = dfd.apply (lambda row: ','.join (map (str, row)), axis=1)
#             #print(dfd)
#             # #print(dfd)
#             # df = df.reset_index ()
#             # df.columns[0] = 'New_ID'
#             # df['New_ID'] = df.index + 880
#             dfd.insert (0, 'clus', range (0, 0 + len (dfd)))
#
#             #
#             # dfd = dfd.reset_index ()
#             # dfd.columns[0] = 'clus'
#             # dfd['clus'] = dfd.index + 0
#             # dfd.index.name = 'clus'
#             # dfd['clus'] = df.index
#             # dfd = dfd.reset_index (drop=True)
#             dfd.to_csv('ozzy.csv')
#             #prediction = model.predict (X)
#             cols = {'clus': model.predict (X),
#                     'url': ddurl}
#             dfda = pd.DataFrame.from_dict (cols)
#             print('okey')
#
#
#
#             dff = dff[['readable', 'reputationHigh', 'reputationLow', 'pictures', 'polarity', 'latency', 'subjectivity',
#                      'words', 'maturity', 'url']]
#             newX = dff.values
#             newX = np.delete (newX, [9], axis=1)
#             result = pickle_model.predict (newX)
#             px2 = result.reshape ((-1, 8))
#             dffres = pd.DataFrame (
#                 {'complete': px2[:, 0], 'accuracy': px2[:, 1], 'precise': px2[:, 2], 'readable': px2[:, 3],
#                  'relevant': px2[:, 4], 'trustworthy': px2[:, 5], 'overall': px2[:, 6], 'neutral': px2[:, 7],
#                     'url': ddurl})
#
#             dfs = [dff, dfda, dffres]
#             dfk = functools.reduce (lambda left, right: pd.merge (left, right, on='url'), dfs)
#             geheel = dfd.merge (dfk, on='clus', how='inner')
#             #print('ozzy')(classes=["table-bordered", "table-striped", "table-hover"])
#             return render_template('adet.html', dataframe=geheel.to_html(index=False))
#             #return render_template ('adet.html', dataframe=dff.to_html (index=False, classes=["table table-sm"]))
#         except:
#             #print(e)
#             flash ('We failed at finding matching URL Please try another Query', 'nameError')
#             return redirect ('/')



@app.route('/getdetail', methods=['POST', 'GET'])
def get_detail():
    if request.method == 'POST':
        try:
            tag = request.form['srch-term']
            tt = tag

            mg = MagicGoogle()
            lijst = []
            #tt = 'Donald Trump'
            search = str(tt+' language:english file:html')
            for url in mg.search_url(query=search):
                lijst.append(url)

            df = pd.DataFrame({'url': lijst})
            #print('parallel versionOzzy: ')
            dff = ((applyParallel(df.groupby(df.index), tmpFunc)))
            dff = dff.query ('words != "err" & latency != "err"')
            #dff.to_csv('tect.csv')
            documents = dff['text'].tolist ()
            #documents.to_csv('t')
            dftekst= dff[['text', 'url']]
            #dftekst.to_csv('ozzy.csv')
            dftekst = dftekst.replace ('\n', ' ', regex=True)
            #print(d)
            ddurl = dff['url'].tolist ()
            #dfurl = dff[['url']].copy()
            #documents = dff['ur'].tolist ()
            #print(documents)
            vectorizer = TfidfVectorizer (stop_words='english')
            X = vectorizer.fit_transform (documents)

            true_k = 4
            model = KMeans (n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
            model.fit (X)

            order_centroids = model.cluster_centers_.argsort ()[:, ::-1]
            terms = vectorizer.get_feature_names ()
            jk=[]
            for i in range (true_k):
                top_ten_words = [terms[ind] for ind in order_centroids[i, :10]]
                j=(' '.join (top_ten_words))
                #print(j)
                jk.append(j)
                #print ("Cluster {}: {}".format (i, ' '.join (top_ten_words)))


            columns=['clus','category']
            dfd = pd.DataFrame(jk)

            dfd.insert (0, 'clus', range (0, 0 + len (dfd)))
            dfd.columns = columns
            #print(dfd)
            se = pd.Series (model.labels_)
            dff = dff[['readable', 'reputationHigh', 'reputationLow', 'pictures', 'polarity', 'latency', 'subjectivity',
                     'words', 'maturity', 'url']]
            newX = dff.values
            newX = np.delete (newX, [9], axis=1)
            result = pickle_model.predict (newX)
            #print('res')
            px2 = result.reshape ((-1, 8))
            dffres = pd.DataFrame (
                {'complete': px2[:, 0], 'accuracy': px2[:, 1], 'precise': px2[:, 2], 'readable': px2[:, 3],
                 'relevant': px2[:, 4], 'trustworthy': px2[:, 5], 'overall': px2[:, 6], 'neutral': px2[:, 7],
                    'url': ddurl ,
                    'clus': se.values})

            #dfs = [dff, dfda, dffres]
            dfs = [dff, dffres]
            dfk = functools.reduce (lambda left, right: pd.merge (left, right, on='url'), dfs)
            geheel = dfd.merge (dfk, on='clus', how='inner')
            #geheel.to_csv('ozkan.csv', index=False)
            #print('ozzy')(classes=["table-bordered", "table-striped", "table-hover"])
            return render_template('adet.html', dataframe=geheel.to_html(index=False))
            #return render_template ('adet.html', dataframe=dff.to_html (index=False, classes=["table table-sm"]))
        except:
            #print(e)
            flash ('We failed at finding matching URL Please try another Query', 'nameError')
            return redirect ('/')
@app.route('/')
def home():
	return render_template('index.html')	

@app.route('/doc')
def doc():
	return render_template('doc.html')

@app.route('/term')
def term():
	return render_template('term.html')

@app.route('/pres')
def pres():
    return redirect(url_for('static', filename='pres.html'))

@app.route('/license')
def license():
    return render_template ('license.html')

def enkele(url):
            #url=row.url
            result_queue = Queue()

            # One Thread for response time
            t1 = Thread(target=f1, args=(result_queue, url))
            t2 = Thread(target=f2, args=(result_queue, url))
            t3 = Thread(target=f3, args=(result_queue, url))
            t4 = Thread(target=f4, args=(result_queue, url))
            #t5 = Thread(target=f5, args=(result_queue, url))
            # Starting threads...
            #print("Start: %s" % time.ctime())
            t1.start()
            #t5.start()
            t2.start()
            t3.start()
            t4.start()

            # Waiting for threads to finish execution...
            t1.join(4)
            #t5.join(4)
            t2.join(4)
            t3.join(4)
            t4.join(4)
    #t.join()
            #print("End:   %s" % time.ctime())

            # After threads are done, we can read results from the queue.
            if not result_queue.empty():
                try:
                    r2 = result_queue.get(f2)
                    r1 = result_queue.get(f1)
                    r3 = result_queue.get(f3)
                    r4 = result_queue.get(f4)
                    dfs = [r1, r2, r3, r4]
                    df = functools.reduce(lambda left, right: pd.merge(left, right, on='url'), dfs)
                except:
                    pass
            #df = enkele ()
            return df
            #df = enkele ()
                #return d

#dff = dff.query('words != "err" & latency != "err" & reputationHigh != "err" & maturity != "err"')


@app.route('/getscore', methods=['POST', 'GET'])
def get_score():
    if request.method == 'POST':
        try:
            tag = request.form['srch-term']
            # print(tag)
            url = str(tag)
            enkele(url)
            #df = ((applyParallel(df.groupby(df.index), tmpFunc)))
            df = (enkele(url))
            #twit = df[['kurtosispolarity','meansentiment','sdpolarity','tweetcount','tweetrate', 'url']]
            dff = df[['readable', 'reputationHigh', 'reputationLow', 'pictures', 'polarity', 'latency', 'subjectivity',
                     'words', 'maturity', 'url']]
            newX = dff.values
            newX = np.delete(newX, [9], axis=1)
            # pickle_fname = 'pickle.model'
            # pickle_model = pickle.load(open(pickle_fname, 'rb'))
            result = pickle_model.predict(newX)  # print (result)
            px2 = result.reshape((-1, 8))
            dffres = pd.DataFrame(
                {'complete': px2[:, 0], 'accuracy': px2[:, 1], 'precise': px2[:, 2], 'readable': px2[:, 3],
                 'relevant': px2[:, 4], 'trustworthy': px2[:, 5], 'overall': px2[:, 6], 'neutral': px2[:, 7]})
            return render_template('mp.html', dataframe=dff.to_html(index=False), res=dffres.to_html(index=False))
        except:
            flash ('This URL is not supported', 'nameError')
            return redirect ('/')


# @app.route('/getscore', methods=['POST', 'GET'])
# def get_score():
#     if request.method == 'POST':
#         tag = request.form['srch-term']
#         #print(tag)
#         url=str(tag)
#         article = Article(url)
#         article.download()
#         article.parse()
#         text = article.text
#         blob = TextBlob ( text )
#         s = Textatistic ( text )
#         vals = requests.get(url, timeout=4, allow_redirects=False).elapsed.total_seconds()
#         st = "/&callback=process&key=57bf606e01a24537ac906a86dc27891f94a0f587"
#         # zz = urlopen ( url )
#         quez = 'http://api.mywot.com/0.4/public_link_json2?hosts=' + url + st
#         stt = urllib.request.urlopen(quez).read()
#         stt = str(stt)
#         wot = re.findall('\d+', stt)
#         ##z=[[conv(s) for s in line.split()] for line in wot]
#         z = [conv(s) for s in wot]
#         high = (z[1])
#         low = (z[2])        #print ( high , low )
#         # WAYBACK
#         zz = "{0.scheme}://{0.netloc}/".format(urlsplit(url))
#         zurlz = "https://web.archive.org/web/0/" + str(zz)
#         r = requests.get(zurlz, allow_redirects=False)
#         data = r.content
#         years = re.findall('\d+', str(data))
#         years = [conv(s) for s in years]
#         years = (years[0])
#         years = int(str(years)[:4])
#         afb = len(article.images)
#         tp = keywords(text, words=3, lemmatize=True)
#         cols = {'maturity': [ years ] ,
#                 'reputationLow': [ low ] ,
#                 'reputationHigh': [ high ] ,
#                 'latency': [ vals ] ,
#                 'words': [ s.word_count ] ,
#                 'subjectivity': [ blob.sentiment.subjectivity ],
#                 'polarity': [ blob.sentiment.polarity ] ,
#                 'pictures': [afb],
#                 'readable': [ s.flesch_score ],
#                 #'kw': [ kw ] ,
#                 'url': [ url ]}
#         dfeat = pd.DataFrame.from_dict ( cols )
#         df = dfeat[['readable', 'reputationHigh', 'reputationLow', 'pictures', 'polarity', 'latency', 'subjectivity', 'url',
#                  'words', 'maturity']]
#         #print(df)
#         #df.to_csv ( 'ft.csv' , index=False , sep=',' , encoding='utf-8' )
#         #del dfeat[ 'url' ]
#         #print (df)
#         newX = df.values
#         # newX=np.delete(newX, [1, 3], axis=1)
#         newX = np.delete(newX, [7], axis=1)
#         # print(newX)
#         #newX = newX[~np.isnan(newX).any(axis=1)]
#         #newX = newX.as_matrix().astype(np.float)
#         pickle_fname = 'pickle.model'
#         pickle_model = pickle.load(open(pickle_fname, 'rb'))
#         result = pickle_model.predict(newX)  # print (result)
#         px2 = result.reshape((-1, 8))
#         dfres = pd.DataFrame(
#             {'complete': px2[:, 0], 'accuracy': px2[:, 1], 'precise': px2[:, 2], 'readable': px2[:, 3],
#              'relevant': px2[:, 4], 'trustworthy': px2[:, 5], 'overall': px2[:, 6], 'neutral': px2[:, 7]})
#
#         #tp = keywords(text, words=3, lemmatize=True)
#         tz =str(tp)
#         tz = re.sub(r"\r\n", " ", tz)
#         print(tz)
#         twtext = list ( )
#         polar = list ( )
#         datum = list ( )
#         for tweet in query_tweets ( tz , 10 ):
#             try:
#                 txt = tweet.text
#                 txt = re.sub ( r"http\S+" , "" , txt )
#                 dat = tweet.timestamp
#                 tblob = TextBlob ( txt )
#                 tpol = tblob.sentiment.polarity
#                 tal = tblob.detect_language()
#                 if tal == ('en'):
#                     twtext.append ( txt )
#                     polar.append ( tpol )
#                     datum.append ( dat )
#                 else:
#                     pass
#             except:
#                 pass
#
#
#         df = pd.DataFrame ( {'tweet': twtext , 'timestamp': datum , 'polarity': polar} )
#         df[ 'timestamp' ] = pd.to_datetime ( df[ 'timestamp' ] )
#         oldest = df[ 'timestamp' ].min ( )
#         newest = df[ 'timestamp' ].max ( )
#         total = (oldest - newest).total_seconds ( )
#         gem = total / len ( df.index )
#         #df.to_csv ( 'sentiment.csv' , index=False , sep=',' , encoding='utf-8' )
#         tmean = df[ "polarity" ].mean ( )
#         tsd = df[ "polarity" ].std ( )
#         tkur = df[ "polarity" ].kurtosis ( )
#         ctweets = {'meansentiment': [ tmean ] ,
#                 'sdpolarity': [ tsd ] ,
#                 'kurtosispolarity': [ tkur ] ,
#                 'tweetrate': [ gem ] ,
#                 'tweetcount': [ len ( df.index ) ] }
#         dftwit = pd.DataFrame.from_dict ( ctweets )
#         #entit
#         my_sent = text
#         parse_tree = nltk.ne_chunk ( nltk.tag.pos_tag ( my_sent.split ( ) ) , binary=True )  # POS tagging before chunking!
#         named_entities = [ ]
#         for t in parse_tree.subtrees ( ):
#             if t.label ( ) == 'NE':
#                 named_entities.append ( t )
#         z = named_entities
#         my_count = pd.Series ( z ).value_counts ( )
#         df = pd.DataFrame ( my_count )
#         df.columns = [ 'Count' ]
#         df[ 'entity' ] = df.index
#         za = df.assign ( entity=[ ', '.join ( [ x[ 0 ] for x in r ] ) for r in df.entity ] )
#         df[ 'entities' ] = pd.DataFrame ( za[ 'entity' ] )
#         del df[ 'entity' ]
#         # tp = str ( keywords ( var_input , words=2 ) )
#         tijd = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
#         col2 = {
#                 #'summary': [summarize(text, words=300)],
#                 'topics': [ tz ] ,
#                  'tittle': [ article.title ] ,
#                  #'published': [ article.publish_date ] ,
#                  #'authors': [ article.authors ] ,
#                  'timestamp(gmtime)': [ tijd ] ,
#                 'url': [ url ]}
#         df2 = pd.DataFrame.from_dict ( col2 )
#         col2 = {
#             'summary': [summarize(text, ratio=0.005)],
#             'url': [url]}
#         dfs = pd.DataFrame.from_dict(col2)
#         #df2.to_csv('scores.csv', index=False)
#     return render_template('tabs.html', dataframe=dfeat.to_html(index=False), dsamen=dfs.to_html(index=False) , res=dfres.to_html(index=False), twit=dftwit.to_html(index=False), ent=df.to_html(index=False), des=df2.to_html(index=False), tag=tag)

    
#@app.route('/article')
#def article():
	#return tory(app.root_path + '/../static/', filename)    



#
@app.route('/article') # this is a job for GET, not POST
def article():
	return send_file('static/notfinalized.pdf',
                     mimetype='application/pdf',
                     attachment_filename='notfinalized.pdf',
                     as_attachment=True)	


@app.route('/feedback')
def index():
    OverallQuality_list = ['1', '2', '3', '4', '5']
    accuracy_list = ['1', '2', '3', '4', '5']
    completeness_list = ['1', '2', '3', '4', '5']
    neutrality_list = ['1', '2', '3', '4', '5']
    precision_list = ['1', '2', '3', '4', '5']
    readibility_list = ['1', '2', '3', '4', '5']
    relevance_list = ['1', '2', '3', '4', '5']
    trustworthiness_list = ['1', '2', '3', '4', '5']
    
    return render_template('feedback.html', OverallQuality_list=OverallQuality_list, accuracy_list=accuracy_list, completeness_list=completeness_list, neutrality_list=neutrality_list, precision_list=precision_list, readibility_list=readibility_list, relevance_list=relevance_list, trustworthiness_list=trustworthiness_list)

@app.route('/create', methods=['POST'])
def create_user():
    if request.form['name'] == '':
        flash('Name cannot be blank', 'nameError')
        return redirect('/feedback')
    if request.form['comment'] == '':
        flash('Comment cannot be blank', 'commentError')
        return redirect('//feedback')
    session['comment'] = request.form['comment']
    comment = countLetters(session['comment'])
    print (comment)
    if comment > 120:
        flash('Not more than 120 characters please', 'commentError')
        return redirect('//feedback')

    session['name'] = request.form['name']
    session['OverallQuality'] = request.form['OverallQuality']
    session['accuracy'] = request.form['accuracy']
    session['completeness'] = request.form['completeness']
    session['neutrality'] = request.form['neutrality']
    session['precision'] = request.form['precision']
    session['readibility'] = request.form['readibility']
    session['relevance'] = request.form['relevance']
    session['trustworthiness'] = request.form['trustworthiness']
    tijd = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
       # tijd = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    col2 = {'OverallQuality': (session['OverallQuality']),
             'accuracy': (session['accuracy']) ,
             'completeness': (session['completeness']) ,
             'neutrality':(session['neutrality']) ,
             'precision': (session['precision']) ,
             'readibility': (session['readibility']) ,
             'relevance': (session['relevance']) ,
             'trustworthiness': (session['trustworthiness']),
             'comment': (session['comment']),             
                'timestamp(gmtime)': [ tijd ]}
    fe = pd.DataFrame.from_dict ( col2 )
    if not os.path.isfile('feed.csv'):
        fe.to_csv('feed.csv', index=False)
    else: # else it exists so append without writing the header
        fe.to_csv('feed.csv',mode = 'a',header=False, index=False)
    return redirect('/process')



        

if __name__ == '__main__':
#~ #    app.run(debug=True)
#    app.run(threaded=True, host="0.0.0.0", port=80)
#s	app.run(processes=3)

#    app.run(host='0.0.0.0', port=80)   #app.run()


        

#if __name__ == '__main__':
#    app.run(debug=True)
#    app.run(threaded=True)
    app.run(host='127.0.0.1')
    #app.run(host='0.0.0.0')




