from __future__ import print_function
import numpy as np
import pandas as pd 
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import re
import os 
import codecs
from sklearn import feature_extraction
#import mpld3


'''
'titles': the titles of the films in their rank order
'synopses': the synopses of the films matched to the 'titles' order
 primary importance is the 'synopses' list; 'titles' is mostly used for labeling purposes.

'''

########  Stopwords, stemming, and tokenizing  ####################################################


# Read movie titles, 100 movies but somehow the last one is empty string
# E:/ds_practice/case_study/document_clustering/data
titles = open('E:/ds_practice/case_study/document_clustering/data/title_list.txt').read().split('\n')
print (type(titles))
print (len(titles))
#print titles[0]
titles = titles[:100]

# Read Genres information

genres = open('E:/ds_practice/case_study/document_clustering/data/genres_list.txt').read().split('\n')
print (type(genres))
print (len(genres))
genres = genres[:100]
#print genres[0]

# Read in the synopses from wiki

synopses_wiki = open('E:/ds_practice/case_study/document_clustering/data/synopses_list_wiki.txt' ,encoding="utf8").read().split('\n BREAKS HERE')
print (type(synopses_wiki))
print (len(synopses_wiki))
synopses_wiki = synopses_wiki[:100]
#print synopses_wiki[0]

# strips html formatting and converts to unicode

synopses_clean_wiki = []
for text in synopses_wiki:
    text = BeautifulSoup(text, 'html.parser').getText()
    synopses_clean_wiki.append(text)
synopses_wiki = synopses_clean_wiki

#print synopses_wiki[0]

# Read synopses information from imdb, which might be different from wiki. Also cleaned as above.

synopses_imdb = open('E:/ds_practice/case_study/document_clustering/data/synopses_list_imdb.txt').read().split('\n BREAKS HERE')
synopses_imdb = synopses_imdb[:100]

synopses_clean_imdb = []

for text in synopses_imdb:
    text = BeautifulSoup(text, 'html.parser').getText()
    synopses_clean_imdb.append(text)
synopses_imdb = synopses_clean_imdb
print (synopses_imdb[0])

# Combine synopses from wiki and imdb

synopses = []
for i in range(len(synopses_wiki)):
    item = synopses_wiki[i] + synopses_imdb[i]
    synopses.append(item)

print (synopses[0])


print(str(len(titles)) + ' titles')
print(str(len(genres)) + ' genres')
print(str(len(synopses)) + ' synopses')


# generates index for each item in the corpora (in this case it's just rank) and I'll use this for scoring later
# the movies in the list are already ranked from 1 to 100

ranks = []
for i in range(1, len(titles)+1):
    ranks.append(i)
    
# load nltk's English stopwords as variable called 'stopwords'
# use nltk.download() to install the corpus first
# Stop Words are words which do not contain important significance to be used in Search Queries


    
stopwords = nltk.corpus.stopwords.words('english')

# load nltk's SnowballStemmer as variabled 'stemmer'

'''
#Snowball is a small string processing language designed for creating 
stemming algorithms for use in 
#Information Retrieval. This site describes Snowball, 
and presents several useful stemmers 
#which have been implemented using it. 
#
#Stemming is just the process of breaking a word down into its root.
'''

stemmer = SnowballStemmer("english")

print (len(stopwords))
print (stopwords)
'''
Today (May 19, 2016) is his only daughter's wedding.
 Vito Corleone is the Godfather. Vito's youngest son, Michael, 
 in a Marine Corps uniform, introduces his girlfriend, 
 Kay Adams, to his family at the sprawling reception.
'''

sents = [sent for sent in nltk.sent_tokenize("Today (May 19, 2016) is his only daughter's wedding. Vito Corleone is the Godfather. Vito's youngest son, Michael, in a Marine Corps uniform, introduces his girlfriend, Kay Adams, to his family at the sprawling reception.")]

print (type(sents))
print(sents[0:2])


words = [word for word in nltk.word_tokenize(sents[0])]
print (type(words))
print (words)

# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
filtered_words = []
for word in words:
        if re.search('[a-zA-Z]', word):
            filtered_words.append(word)
print (type(filtered_words))
print ('********************************************************************************')
print (filtered_words)

# see how "only" is stemmed to "onli" and "wedding" is stemmed to "wed"
stems = [stemmer.stem(t) for t in filtered_words]
print ('***********************************************************')
print (stems)

'''
#two functions:
#
#tokenize_and_stem: tokenizes (splits the synopsis into a list of its respective words (or tokens) and also stems 
#each token 
#tokenize_only: tokenizes the synopsis only
#
#use both these functions to create a dictionary which becomes important in case I want to use stems for an algorithm,
#but later convert stems back to their full words for presentation purposes.
'''

# tokenizer and stemmer which returns the set of stems in the text that it is passed
# Punkt Sentence Tokenizer, sent means sentence

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

# Sample call
words_stemmed = tokenize_and_stem("Today (May 19, 2016) is his only daughter's wedding.")
print ('****************************************************')
print (words_stemmed)

words_only = tokenize_only("Today (May 19, 2016) is his only daughter's wedding.")
print ('******************************************************')
print (words_only)


# Below We use our stemming/tokenizing and tokenizing functions to iterate over the list of synopses
# to create two vocabularies: one stemmed and one only tokenized

# extend vs. append
a = [1, 2]
b = [3, 4]
c = [5, 6]
b.append(a)
c.extend(a)
print(b)
print(c)

'''
#Below we use  stemming/tokenizing and tokenizing functions to iterate over the list of synopses 
#to create two vocabularies: one stemmed and one only tokenized.
'''


totalvocab_stemmed = []
totalvocab_tokenized = []
for i in synopses:
    allwords_stemmed = tokenize_and_stem(i) # for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) # extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
    
    
print(len(totalvocab_stemmed))
print(len(totalvocab_tokenized))

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
print(vocab_frame.head())

words_frame = pd.DataFrame({'WORD': words_only}, index = words_stemmed)
print('there are ' + str(words_frame.shape[0]) + ' items in words_frame')
print(words_frame)


#  Tf-idf and document similarity

'''
#Here, term frequency-inverse document frequency (tf-idf) 
vectorizer parameters and then
#convert the synopses list into a tf-idf matrix.
#
#To get a Tf-idf matrix, first count word occurrences by document. 
This is transformed into a 
#document-term matrix (dtm). 
This is also just called a term frequency matrix. 
#An example of a dtm is:
#    
#Then apply the term frequency-inverse document frequency weighting: words that occur frequently 
#within a document but not frequently within the corpus receive a higher weighting as these words 
#are assumed to contain more meaning in relation to the document.
#
#
#A couple things to note about the parameters we define below:
'''

# Note that the result of this block takes a while to show
from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

get_ipython().magic(u'time tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses')

# (100, 563) means the matrix has 100 rows and 563 columns
print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
len(terms)
print(type(terms))
print (terms)

from sklearn.metrics.pairwise import cosine_similarity
# A short example using the sentences above
words_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

get_ipython().magic(u'time words_matrix = words_vectorizer.fit_transform(sents) #fit the vectorizer to synopses')

# (2, 18) means the matrix has 2 rows (two sentences) and 18 columns (18 terms)
print(words_matrix.shape)
print(words_matrix)

# this is how we get the 18 terms
analyze = words_vectorizer.build_analyzer()
print(analyze("Today (May 19, 2016) is his only daughter's wedding."))
print(analyze("Vito Corleone is the Godfather."))
print(analyze("Vito's youngest son, Michael, in a Marine Corps uniform, introduces his girlfriend, Kay Adams, to his family at the sprawling reception."))
all_terms = words_vectorizer.get_feature_names()
print(all_terms)
print(len(all_terms))

# sent 1 and 2, similarity 0, sent 1 and 3 shares "his", sent 2 and 3 shares Vito - try to change Vito's in sent3 to His and see the similary matrix changes
example_similarity = cosine_similarity(words_matrix)
example_similarity


# Now onto the fun part. Using the tf-idf matrix, you can run a slew of clustering algorithms to better understand the hidden structure within the synopses. I first chose k-means. K-means initializes with a pre-determined number of clusters (I chose 5). Each observation is assigned to a cluster (cluster assignment) so as to minimize the within cluster sum of squares. Next, the mean of the clustered observations is calculated and used as the new cluster centroid. Then, observations are reassigned to clusters and centroids recalculated in an iterative process until the algorithm reaches convergence.
# 
# I found it took several runs for the algorithm to converge a global optimum as k-means is susceptible to reaching local optima - how to decide that the algorithm converged???

from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters)

get_ipython().magic(u'time km.fit(tfidf_matrix)')

clusters = km.labels_.tolist()

print (clusters)

# We use joblib.dump to pickle the model, once it has converged and to reload the model/reassign the labels as the clusters.
from sklearn.externals import joblib

#uncomment the below to save your model 
#since I've already run my model I am loading from the pickle

joblib.dump(km,  'doc_cluster.pkl')

km = joblib.load('E:/ds_practice/case_study/document_clustering/doc_cluster.pkl')
clusters = km.labels_.tolist()
# clusters show which cluster (0-4) each of the 100 synoposes belongs to
print(len(clusters))
print(clusters)

# Here, I create a dictionary of titles, 
#vranks, the synopsis, the cluster assignment, 
# and the genre [rank and genre were scraped from IMDB].

films = { 'title': titles, 'rank': ranks, 'synopsis': synopses, 'cluster': clusters, 'genre': genres }

frame = pd.DataFrame(films, index = [clusters] , columns = ['rank', 'title', 'cluster', 'genre'])

print(frame) # here the ranking is still 0 to 99
frame.to_excel('keman.xlsx')

frame['cluster'].value_counts() #number of films per cluster (clusters from 0 to 4)

grouped = frame['rank'].groupby(frame['cluster']) # groupby cluster for aggregation purposes

print (grouped.mean()) # average rank (1 to 100) per cluster



# Note that clusters 4 and 0 have the lowest rank, which indicates that they, on average, contain films that were ranked as "better" on the top 100 list.
# Here is some fancy indexing and sorting on each cluster to identify which are the top n (I chose n=6) words that are nearest to the cluster centroid. 
# This gives a good sense of the main topic of the cluster.




print("Top terms per cluster:")

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace
    
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace

# Cosine similarity is measured against the tf-idf matrix and can be used to generate a measure of similarity between each document and the other documents 
# in the corpus (each synopsis among the synopses). cosine similarity 1 means the same document, 0 means totally different ones. 
# dist is defined as 1 - the cosine similarity of each document.  
# Subtracting it from 1 provides cosine distance which I will use for plotting on a euclidean (2-dimensional) plane.
# Note that with dist it is possible to evaluate the similarity of any two or more synopses.
    
    
similarity_distance = 1 - cosine_similarity(tfidf_matrix)
print(type(similarity_distance))
print(similarity_distance.shape)

# Multidimensional scaling
# Here is some code to convert the dist matrix into a 2-dimensional array using multidimensional scaling. 
# I won't pretend I know a ton about MDS, but it was useful for this purpose. 
# Another option would be to use principal component analysis.

import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

get_ipython().magic(u'time pos = mds.fit_transform(similarity_distance)  # shape (n_components, n_samples)')

print(pos.shape)
print(pos)

xs, ys = pos[:, 0], pos[:, 1]
print(type(xs))
xs

# Visualizing document clusters
# In this section, I demonstrate how you can visualize the document clustering output using matplotlib and mpld3 (a matplotlib wrapper for D3.js).
# First I define some dictionaries for going from cluster number to color and to cluster name. 
# I based the cluster names off the words that were closest to each cluster centroid.

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

#set up cluster names using a dict
cluster_names = {0: 'Family, home, war', 
                 1: 'Police, killed, murders', 
                 2: 'Father, New York, brothers', 
                 3: 'Dance, singing, love', 
                 4: 'Killed, soldiers, captain'}


# Next, I plot the labeled observations (films, film titles) colored by cluster using matplotlib. 
# I won't get into too much detail about the matplotlib plot, but I tried to provide some helpful commenting.

#some ipython magic to show the matplotlib plots inline
get_ipython().magic(u'matplotlib inline')

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 

print(df[1:10])
# group by cluster
# this generate {name:group(which is a dataframe)}
groups = df.groupby('label')
print(groups.groups)


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
# ms: marker size
for name, group in groups:
    print("*******")
    print("group name " + str(name))
    print(group)
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=20, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=10)  

    
    
plt.show() #show the plot

#uncomment the below to save the plot if need be
#plt.savefig('clusters_small_noaxes.png', dpi=200)


# Use plotly to generate interactive chart. I have to downgrade matplotlib to 1.3.1 for this chart to work with plotly. see https://github.com/harrywang/plotly/blob/master/README.md for how to setup plotly. After running the following, a browser will open to show the plotly chart.

#import plotly.plotly as py
#plot_url = py.plot_mpl(fig)









