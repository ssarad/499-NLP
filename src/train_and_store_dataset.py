import nltk
import collections
import os 
import sys 
import extract as dt
import algorithm
from nltk import chunk #NLTK trees

""" Source -> https://gist.github.com/japerk/1909413 """
def data():
    trainer = "../data/tokenized/in_domain_train.tsv"
    return dt.extractFromDataset((dt.getLinesFromFile(trainer)))

def get_sents():
    dataset = data()
    sents = []
    for items in dataset:
        sents.append(items[3])
    return sents

def train():
    for items in sentences_only:
        algorithm.update_gram(items)
    #Test sentences
    test = ["Javier likes carrots", "Julie is annoyed and sad", "Harsha said he likes Spain, Italty and Greece", "Zeus is a Greek god, is he?"]
    algorithm.parser()


def append_to_tree():
    sent_with_trees = []
    for items in get_sents():
        sent_with_trees.append(chunk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(items))))
    #return chunk.ne_chunk(tagged_sent) #POS Taggers
    return sent_with_trees