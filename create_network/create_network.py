import numpy as np
import networkx as nx
import gensim
from itertools import combinations

def lda_setup(tokenized, min_words=0):
    dictionary = gensim.corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=200)
    words = set(dictionary.values())
    if min_words == 0:
        BoW = [ dictionary.doc2bow(doc, allow_update=False) for doc in tokenized ]
        return dictionary, BoW
    else:
        BoW, indices = list(), list()
        for n, doc in enumerate(tokenized):
            if len(set(doc) & words) >= min_words:
                BoW.append( dictionary.doc2bow(doc, allow_update=False) )
                indices.append(n)
        return dictionary, BoW, indices

def BoW_matrix(BoW, dictionary):
    dense = gensim.matutils.corpus2dense(BoW, len(dictionary))
    return dense / dense.sum(axis=0, keepdims=True)

## create linkage graph
def linkage_graph(dictionary, dense_norm, log = True):
    G = nx.Graph()
    num_docs = np.shape(dense_norm)[1]
    dense_word = dense_norm.sum(axis=1)
    for i, j in combinations(dictionary.keys(), 2):

        X0 = dense_norm[i]
        X1 = dense_norm[j]

        link = np.dot(X0, X1) / ( dense_word[i] * dense_word[j] ) * num_docs
        if log == False:
            if link > 0:
                G.add_edge(dictionary[i], dictionary[j], weight = link)
        else:
            if link > 0:
                G.add_edge(dictionary[i], dictionary[j], weight = np.log2(link))
    return G

def get_network(docs, bigram_generator, arg_words='tc_1200.txt', filename='CMV'):
    if type(docs) == str:
        docs = gensim.models.word2vec.LineSentence(docs)
    else:
        pass

    if type(bigram_generator) == str:
        bigram_generator = gensim.models.phrases.Phrases.load(bigram_generator)
    else:
        pass

    # import 1200 phrases (unigrams+bigrams) from LIWC's tentative/certain list
    if type(arg_words) == str:
        with open(arg_words, 'r') as file:
            arg_words = file.readline().split()

    # create a text file filtered with those 1200 phrases
    with open(filename+'_texttc.txt', 'w') as file:
        for doc in bigram_generator[docs]:
            for word in doc:
                if word in arg_words:
                    file.write(word+' ')
            file.write('\n')

    # create and save the network where link is the pairwise mutual information
    docs_tcfilt = gensim.models.word2vec.LineSentence(filename+'_texttc.txt')
    dict_tcfilt, BoW_tcfilt, ind_tcfilt = lda_setup(docs_tcfilt, min_words=2)
    mat_tcfilt = BoW_matrix(BoW_tcfilt, dict_tcfilt)
    G = linkage_graph(dict_tcfilt, mat_tcfilt, log = True)
    nx.write_gexf(G, filename+'_backbone_network.gexf')

    return G
