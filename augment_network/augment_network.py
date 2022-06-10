import pickle

import re
import regex as reg

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from sknetwork.clustering import bimodularity
import gensim

def findall_words(string, bi_reg, bi_re, uni_re):
    string = ' '+string+' '
    string = string.replace(' ', '  ') # two spaces so that we catch everything

    bis = reg.findall(bi_reg, string, overlapped=True) # include both 'a b' and 'b c'
    string = re.sub(bi_re, '', string)
    unis = re.findall(uni_re, string)
    string = re.sub(uni_re, '', string)

    return bis+unis # return semantic and pronoun count

def tc_cluster_linkage(mat_coarse, mat_tc, dict_tc, cluster_df):
    # for all matrices: columns must be documents

    num_clus = len(set(cluster_df['modularity_class']))
    values = set(dict_tc.values())

    word_link = dict()

    for word in values:
        word_underbar = word[1:-1].replace('  ', '_')
        mat_word = mat_tc[dict_tc.token2id[word]].A[0]

        mat_coarse[cluster_df['modularity_class'][word_underbar]] -= mat_word

        ind_null = np.where(mat_coarse.sum(axis=0)>0)[0]
        mat_word_filt = mat_word[ind_null] # only the documents with indices
        mat_coarse_filt = mat_coarse[:, ind_null]

        # eliminate word from cluster coarse matrix
        cluster_count = mat_coarse_filt.sum(axis=1)
        pc = cluster_count / cluster_count.sum()

        cluster_norm = mat_coarse_filt / mat_coarse_filt.sum(axis=0, keepdims=True)

        doc_with_word = np.where(mat_word_filt != 0)[0] # numpy matrix should have index 0
        word_count_filt = mat_word_filt[doc_with_word]

        cluster_norm_filt = cluster_norm[:, doc_with_word]

        pcw_raw = np.sum(cluster_norm_filt * np.tile(word_count_filt, (num_clus, 1)), axis=1)
        pcw = pcw_raw / pcw_raw.sum()

        links = list()

        for i in range(num_clus):
            if pcw[i] > 0:
                link = pcw[i] / pc[i]
                links.append( np.log2(link) )
            else:
                links.append(np.nan)
        word_link[word] = links

        mat_coarse[cluster_df['modularity_class'][word_underbar]] += mat_word # reset tc_coarse

    return pd.DataFrame.from_dict(word_link, orient='index')

def word_cluster_linkage(mat_coarse, mat_ext, dict_ext, cluster_df):
    # measure linkage between clusters and each candidate word
    num_clus = len(set(cluster_df['modularity_class']))
    words_ext = set(dict_ext.values())

    ind_null = np.where(mat_coarse.sum(axis=0)>0)[0]

    mat_ext_filt = mat_ext[:, ind_null] # only documents with indices
    mat_coarse_filt = mat_coarse[:, ind_null]
    num_docs = len(ind_null)

    words = set(dict_ext.values())

    word_link = dict()

    cluster_count = mat_coarse_filt.sum(axis=1)
    pc = cluster_count / cluster_count.sum()

    cluster_norm = mat_coarse_filt / mat_coarse_filt.sum(axis=0, keepdims=True)

    for word in words:
        mat_word = mat_ext_filt[dict_ext.token2id[word]].A
        doc_with_word = np.where(mat_word != 0)[1]
        word_count_filt = mat_word[:, doc_with_word]

        cluster_norm_filt = cluster_norm[:, doc_with_word]

        pcw_raw = np.sum(cluster_norm_filt * np.tile(word_count_filt, (num_clus, 1)), axis=1)
        pcw = pcw_raw / pcw_raw.sum()

        links = list()

        for i in range(num_clus):
            if pcw[i] > 0:
                link = pcw[i] / pc[i]
                links.append( np.log2(link) )
            else:
                links.append(np.nan)
        word_link[word] = links
    return pd.DataFrame.from_dict(word_link, orient='index')

def filter_words(cluster_df, docs, filename, candidate_words):
    # cluster_df must be a form of dataframe
    if type(docs) == str:
        with open(docs, 'r') as file:
            docs = file.read().splitlines()
    else:
        pass

    if type(candidate_words) == str:
        with open(candidate_words, 'r') as file:
            candidate_words = file.readline().split()
    else:
        pass

    bigrams = ''
    unigrams = ''

    augmented = list(cluster_df.index) + candidate_words
    augmented.sort()

    for w in augmented:
        if '_' in w:
            bigrams += '| ' + w.replace('_','  ') + ' '
        else:
            unigrams += '| ' + w + ' '
    bigrams = bigrams[1:]
    unigrams = unigrams[1:]

    bi_re = re.compile(bigrams)
    bi_reg = reg.compile(bigrams)
    uni_re = re.compile(unigrams)

    # filter words (could take a while depending on the size of the data)
    hit_words = list()
    for doc in docs:
        hit_word = findall_words(string=doc, bi_reg=bi_reg, bi_re=bi_re, uni_re=uni_re)
        hit_words.append(hit_word)

    # save it into list
    with open(filename+'_filtered.pickle', 'wb') as handle:
        pickle.dump(hit_words, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return hit_words

def get_augmented_cluster(cluster_assignment_path, candidate_words, filename='CMV', docs=None, filtered_doc_path=None):

    cluster_df = pd.read_csv(cluster_assignment_path, index_col=0)
    num_clus = len(set(cluster_df['modularity_class']))

    if type(docs) == str:
        with open(docs, 'r') as file:
            docs = file.read().splitlines()
    else:
        pass

    if filtered_doc_path==None:
        hit_words = filter_words(cluster_df=cluster_df, docs=docs, filename=filename, candidate_words=candidate_words)
    else:
        with open(filtered_doc_path, 'rb') as handle:
            hit_words = pickle.load(handle)

    # measure linkage between each word and cluster (how strong is each word linked to its cluster?)
    dict_tc = gensim.corpora.Dictionary(hit_words)

    # filter words (only include words from backbone)
    tc_tokens = set(cluster_df.index) & set([ w[1:-1].replace('  ','_') for w in dict_tc.values() ])
    tc_ids = [ dict_tc.token2id[' ' + i.replace('_', '  ') + ' '] for i in tc_tokens ]
    dict_tc.filter_tokens(good_ids=tc_ids)

    BoW_tc = [dict_tc.doc2bow(doc, allow_update=False) for doc in hit_words ]
    mat_tc = gensim.matutils.corpus2csc(BoW_tc, len(dict_tc))

    "create cluster count matrix by adding rows"
    denses = list()
    for i in range(num_clus):
        words = set(cluster_df[cluster_df['modularity_class'] == i].index)
        ids = [ dict_tc.token2id[' ' + w.replace('_', '  ') + ' '] for w in words
        if w in tc_tokens ]
        denses.append(mat_tc[ids, :].sum(axis=0))
    clus_count_tc = np.vstack(denses).A
    "------------------------------------------"

    linkages_tc = tc_cluster_linkage(clus_count_tc, mat_tc, dict_tc, cluster_df)
    # output: dataframe with columns as clusters and indices as words from backbone

    # reassign clusters to maximize modularity of the bipartite backbone graph
    G_tc = nx.bipartite.from_biadjacency_matrix(sparse.bsr_matrix(linkages_tc).T)
    top, bottom = nx.bipartite.sets(G_tc)
    node_name = ['C'+str(n) for n in range(num_clus)] + [ linkages_tc.index[n-num_clus] for n in list(bottom)]
    name_map = {i:node_name[i] for i in G_tc.nodes }
    G_tc = nx.relabel_nodes(G_tc, name_map)

    #c_label = ['C'+str(n) for n in range(num_clus)]
    reassign = [ np.argmax(linkages_tc.loc[ind, :]) for ind in linkages_tc.index ]
    bimod_null = bimodularity(np.nan_to_num(linkages_tc,0).clip(0), labels=np.array(reassign),
                              labels_col=np.array(list(range(num_clus))))

    # creating a matrix that counts candidates
    dict_aug = gensim.corpora.Dictionary(hit_words)
    # filter words (exclude words from backbone)
    dict_aug.filter_tokens(bad_ids=tc_ids)

    BoW_augcand = [dict_aug.doc2bow(doc, allow_update=False) for doc in hit_words ]
    mat_augcand = gensim.matutils.corpus2csc(BoW_augcand, len(dict_aug))

    linkages_cand = word_cluster_linkage(clus_count_tc, mat_augcand, dict_aug, cluster_df)
    # output: dataframe with columns as clusters and indices as words from candidates

    # for each candidate word, measure modularity when the word is added to the bipartite graph
    new_mod = dict()
    for ind in linkages_cand.index:
        newvec = np.array(linkages_cand.loc[ind,:])
        newmat = np.vstack([np.array(linkages_tc), newvec])
        newc = np.argmax(newvec)
        new_mod[ind] = (newc, bimodularity(np.nan_to_num(newmat.clip(0),0),
                             labels=np.array(reassign+[newc]), labels_col=np.array(list(range(num_clus)))))

    # final: add word only if it increases the modularity
    aug_clus = [ (i, j[0]) for i, j in new_mod.items() if j[1] > bimod_null ]

    cluster_aug = [ (i[1:-1].replace('  ','_'),j) for i, j in aug_clus ]
    cluster_reassign = [ (ind[1:-1].replace('  ','_'), np.argmax(linkages_tc.loc[ind, :]))
     for ind in linkages_tc.index ]

    augmented_df = pd.DataFrame(cluster_reassign+cluster_aug, columns=['word', 'modularity_class']
    ).set_index('word')

    augmented_df.to_csv(filename+'_cluster_augmented.csv')

    return augmented_df
    ## augmented cluster created
