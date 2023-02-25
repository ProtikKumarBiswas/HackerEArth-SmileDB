from tqdm import tqdm
from gensim.models import Word2Vec
import numpy as np

class AvgWord2Vec:
    
    def __init__(self):
        self.w2v_model = None
        self.w2v_words = None
        self.w2v_size = 50

    def fit(self, x, vector_size = 50):
        print('Running Average Word2Vec - .fit() .....')
        
        self.w2v_size = vector_size
        
        list_of_sentance=[]
        for sentance in x:
            list_of_sentance.append(sentance.split())
        
        self.w2v_model=Word2Vec(list_of_sentance,min_count=5,size=self.w2v_size, workers=10)
        self.w2v_words = list(self.w2v_model.wv.vocab)
        
        sent_vectors = []; # the avg-w2v for each sentence is stored in this list
        for sent in tqdm(list_of_sentance): # for each sentence
            sent_vec = np.zeros(self.w2v_size)
            cnt_words =0; # num of words with a valid vector in the sentence
            for word in sent: # for each word in a sentence
                if word in self.w2v_words:
                    vec = self.w2v_model.wv[word]
                    sent_vec += vec
                    cnt_words += 1
            if cnt_words != 0:
                sent_vec /= cnt_words
            sent_vectors.append(sent_vec)
        return sent_vectors
        
    
    def transform(self, x):
        print('Running Average Word2Vec - .transform() .....')
        print('Vector Size is set to ', self.w2v_size)
        
        list_of_sentance=[]
        for sentance in x:
            list_of_sentance.append(sentance.split())
        
        sent_vectors = []; # the avg-w2v for each sentence is stored in this list
        for sent in tqdm(list_of_sentance): # for each sentence
            sent_vec = np.zeros(self.w2v_size)
            cnt_words =0; # num of words with a valid vector in the sentence
            for word in sent: # for each word in a sentence
                if word in self.w2v_words:
                    vec = self.w2v_model.wv[word]
                    sent_vec += vec
                    cnt_words += 1
            if cnt_words != 0:
                sent_vec /= cnt_words
            sent_vectors.append(sent_vec)
        return sent_vectors