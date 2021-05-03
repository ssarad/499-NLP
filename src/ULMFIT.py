from fastai
from fastai import *
from fastai.text import *
import pandas as pd 
import numpy 
import functools 
import os 
import io 
import data-extract
import nltk
from sklearn.model_selection import train_test_split 


nltk.download('stopwords')

class ULMFIT():

    def __init__(self):
        file_to_extract = data-extract.extractFromDataset()#Change this to the dataset
        self.stop_words = stopwords.words('english')
        #self.pand = pd.DataFrame({'Text:'file_to_extract})
        self.pand = pd.DataFrame({'Text': []})

    def tokenize(self):
        return self.pand['Text'].apply(lambda x : x.split())

    def remove_stop_words(self):
        return self.tokenize().apply(lambda x : [x for x in data if x not in stop_words])

    def detokenize(self):
        arr = []
        param = self.remove_stop_words()
        for x in range(len(self.pand)):
            x = ' '.join(param[x])
            arr.append(x)
        self.pand['Text'] = arr 
        return self.pand
    
    def train_validation(self):
        #Tune the test size per the dataset, and random state with 20 seems to be the sweet spot
        x,u = train_test_split(self.pand, stratify = self.pand['label'], test_size = 0.2, random_state = 20)
        return x, u
    
    def lang_model(self):
        x,u = self.train_validation()
        lml = TextLMDataBunch.from_df(train_df = x, valid_df = u, path = "") #IMPORTANT ! SET THIS PATH!
        data_lrn = language_model_learner(lml, pretrained_model=URLS.WT103, drop = 0.01) #Adjust drop as per needed, 0.1 should suffice
        return data_lrn
    
    def train(self):
        classification = self.lang_model()
        return classification.one_cycle(1, 1e-2)


