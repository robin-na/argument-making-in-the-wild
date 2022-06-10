# The Diversity of Argument-Making in the Wild: from Assumptions and Definitions to Causation and Anecdote in Reddit’s “Change My View”

Robin W. Na and Simon DeDeo. Proceedings of the $44^{th}$ Annual Conference of the Cognitive Science Society. 2022.
https://arxiv.org/abs/2205.07938

This repository shares the cluster assignment of argument fragments (unigrams and bigrams related to argument-making) used in the original paper along with python codes to construct the network using preprocessed documents. As described below, the repository is divided into 3 folders.

### 1. Cluster Assignment

The folder ```cluster_assignment``` consists of 2 csv files, ```cluster_tc.csv``` and ```cluster_aug.csv``` where the former is one version of various cluster assignments derived from a network created from the second folder ```create_network``` using the modularity clustering algorithm in Gephi. The latter is the augmented version of such assignment, which could be implemented by the thrid folder ```augment_network```.

The column "modularity_class" is indexed into 6 different clusters where

| Index | Pattern Name |
| ----- | ----- |
| 0 | Relevance & Presumption |
| 1 | Definition & Clarity |
| 2 | Deduction & Certainty |
| 3 | Causation & Examples |
| 4 | Induction & Probability |
| 5 | Personal & Anecdotal |

### 2. Create Network

You can make your own lexical network with the same methodology described in the paper in this folder, given any preprocessed text where (1) all symbols are eliminated and every letter is lowercased and (2) each document is divided by a linesplit (if organized into a ```.txt``` file).

In Python, make sure that your directory is set to the ```create_network``` folder and run
```
import create_network
create_network.get_network(docs= your_preprocessed_document , bigram_generator= your_bigram_generator)
```

```your_preprocessed_document``` could either be a list of strings where each element is a preprocessed documents, or a string that describes the path of the document. The original r/ChangeMyView text is available upon request.

```your_bigram_generator``` could be any module of ```gensim.models.phrases.Phrases()``` that would glue appropriate adjacent words into bigrams (e.g., ``` `kind of` ``` to ``` `kind_of` ```). The generator we used for the paper is also available upon request.

By default, the code will refer to ```tc_1200.txt``` in the folder and filter your preprocessed documents with those bigrams and unigrams. You can set your own directory by changing the ```arg_words``` argument in ```get_network()``` function. This will spit out ```[filename]_texttc.txt``` which is the filtered version of your input text data and ```[filename]_backbone_network.gexf``` which is the lexical network with argument fragments as nodes and each pairwise mutual information as link. ```[filename]``` is set to ``` `CMV` ``` by default.

Now that you have the network, you can open this with Gephi or any program you prefer and run a modularity-based clustering algorithm to get cluster assignments. Note that you would want to threshold the link with at least 0 as negative links could make the community detection complicated.


### 3. Augment Network

You can now augment the cluster assignment with the same methodology described in the paper in this folder, given a cluster assignment file with the same format as ```cluster_assignment/cluster_tc.csv``` and preprocessed text which we highly recommend using the same one you used in your ```create_network``` procedure.

In Python, make sure that your directory is set to the ```augment_network``` folder and run
```
import augment_network
augment_network.get_augmented_cluster(cluster_assignment_path='cluster_tc.csv', candidate_words='candidate_words.txt'
                                      , docs=your_preprocessed_document)
```

This will create ```[filename]_filtered.pickle``` which is a list of list where each list is a tokenized document filtered by original words from ``` cluster_tc.csv ``` and candiate words. It will also create ```[filename]_cluster_augmented.csv``` which is the augmented version of cluster assignment. In other words, this function will take a ```cluster_assignment/cluster_tc.csv``` file and return something like ```cluster_assignment/cluster_aug.csv```.

