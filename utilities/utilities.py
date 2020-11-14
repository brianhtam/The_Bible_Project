import spacy
import re
import string
import pandas as pd

# pyLDAvis
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
from sklearn.decomposition import LatentDirichletAllocation

# Import sklearn to do CountVectorizing and TF-IDF document-term matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Spacy tools
# Create our list of punctuation marks
punctuations = string.punctuation
# Load English tokenizer, tagger, parser, NER and word vectors
parser = spacy.load('en_core_web_sm', disable=["parser", "ner"])
# Create our list of stopwords
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# 2D Visuals
from sklearn.decomposition import PCA
from itertools import cycle
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

# Creating our tokenizer function
def names_only(sentence):
    '''
    Strips the sentence to leave only the proper nouns
    '''
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)
    
    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.orth_ for word in mytokens if word.pos_ == 'PROPN']
    
    # return preprocessed list of tokens
    return ' '.join(mytokens)

def spacy_tokenizer(sentence):
    '''
    cleans up the
    '''
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)
    
    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    
    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    
    # alphanumeric
    mytokens = [re.sub('\w*\d\w*', ' ', x) for x in mytokens]
    
    # punc_lower 
    mytokens = [re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower()) for x in mytokens]
    
    # return preprocessed list of tokens
    return ' '.join(mytokens)

# Topic Modeling Utilities

def display_topics(model, feature_names, no_top_words, topic_names=None):
    '''
    Takes in model and feature names and outputs 
    a list of string of the top words from each topic.
    '''
    topics = []
    for ix, topic in enumerate(model.components_):
        topics.append(str(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])))
    return topics

def pyLDAviz(lda, df_tfidf, tfidf):
    return pyLDAvis.sklearn.prepare(lda, df_tfidf, tfidf)

def pyLDAvis_pipeline(df, n_components):
    # Define what you'll feed into the vectorizer as X
    X = df['cleaner']
    tfidf = TfidfVectorizer(stop_words='english')
    df_tfidf = tfidf.fit_transform(X)
    df_tfidf_df = pd.DataFrame(df_tfidf.toarray(), columns=tfidf.get_feature_names())
    print('Vocab size: ', len(df_tfidf_df.columns))
    
    # for TFIDF DTM
    lda = LatentDirichletAllocation(n_components=n_components, random_state=0)
    topic_array = lda.fit_transform(df_tfidf)

    # Topic-term Matrix
    topics = display_topics(lda, tfidf.get_feature_names(), 3)
    topic_word = pd.DataFrame(lda.components_.round(3),
                 index =  topics,
                 columns = tfidf.get_feature_names())

    # Document-topic Matrix
    df_topics = pd.DataFrame(topic_array.round(5),
                 index = X.index,
                 columns = topics)

    return pyLDAvis.sklearn.prepare(lda, df_tfidf, tfidf)

def plot_tsne(df, n_components, target, metric):
    '''
    Plots the TSNE plot showing seperation of. 
    Given:
    df = dataframe of focus
    n_components = number of topics
    target = what to color by
    metric = 'cosine' or 'euclidean' distance?  
    '''
    # Define what you'll feed into the vectorizer as X
    X = df['cleaner']
    tfidf = TfidfVectorizer(stop_words='english')
    df_tfidf = tfidf.fit_transform(X)
    df_tfidf_df = pd.DataFrame(df_tfidf.toarray(), columns=tfidf.get_feature_names())
    print('Vocab size: ', len(df_tfidf_df.columns))
    
    # for TFIDF DTM
    lda = LatentDirichletAllocation(n_components=n_components, random_state=0)
    doc_topic = lda.fit_transform(df_tfidf)
    
    target_names = list(target.unique())

    # fit tsne
    tsne_model = TSNE(n_components=2, random_state=42, metric=metric)
    data = tsne_model.fit_transform(doc_topic)
    
    # plot tsne
    colors = cycle(['blue','orange','red','black','cyan'])
    for c, label in zip(colors, target_names):
        plt.scatter(data[target == label, 0], data[target == label, 1], c=c, label=label, s=0.4, alpha=0.5)
    plt.legend(fontsize=6, loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor='#333333')
    plt.xlim(-100,100);
