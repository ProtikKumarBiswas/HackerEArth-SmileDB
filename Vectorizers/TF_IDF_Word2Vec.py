from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from gensim.models import Word2Vec
import numpy as np

class TF_IDF_Word2Vec:
    def __init__(self):
        self.dictionary = None
        self.tfidf_feat = None
        self.model = None
        self.w2v_model = None
        self.w2v_words = None
        self.w2v_size = 50
        
    def fit(self, x, vector_size = 50):
        print('Running TF-IDF Weighted Word2Vec - .fit() .....')
        
        self.w2v_size = vector_size
        
        list_of_sentance=[]
        for sentance in x:
            list_of_sentance.append(sentance.split())
        
        self.w2v_model=Word2Vec(list_of_sentance,min_count=5,size=self.w2v_size, workers=10)
        self.w2v_words = list(self.w2v_model.wv.vocab)
        
        self.model = TfidfVectorizer()
        tf_idf_matrix = self.model.fit_transform(x)
        # we are converting a dictionary with word as a key, and the idf as a value
        self.dictionary = dict(zip(self.model.get_feature_names(), list(self.model.idf_)))
        
        # TF-IDF weighted Word2Vec
        self.tfidf_feat = self.model.get_feature_names()
        # final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf
        
        tfidf_sent_vectors = [];
        for sent in tqdm(list_of_sentance): # for each sentence 
            sent_vec = np.zeros(self.w2v_size) # as word vectors are of zero length
            weight_sum =0; # num of words with a valid vector in the sentence
            for word in sent: # for each word in a sentence
                if word in self.w2v_words and word in self.tfidf_feat:
                    vec = self.w2v_model.wv[word]
                    tf_idf = self.dictionary[word]*(sent.count(word)/len(sent))
                    sent_vec += (vec * tf_idf)
                    weight_sum += tf_idf
            if weight_sum != 0:
                sent_vec /= weight_sum
            tfidf_sent_vectors.append(sent_vec)
        return tfidf_sent_vectors
        

    
    def transform(self, x):
        print('Running TF-IDF Weighted Word2Vec - .transform() .....')
        print('Vector Size is set to ', self.w2v_size)
        
        list_of_sentance=[]
        for sentance in x:
            list_of_sentance.append(sentance.split())
        
        tf_idf_matrix = self.model.transform(x)
        
        tfidf_sent_vectors = [];
        for sent in tqdm(list_of_sentance): # for each sentence 
            sent_vec = np.zeros(self.w2v_size) # as word vectors are of zero length
            weight_sum =0; # num of words with a valid vector in the sentence
            for word in sent: # for each word in a sentence
                if word in self.w2v_words and word in self.tfidf_feat:
                    vec = self.w2v_model.wv[word]
                    tf_idf = self.dictionary[word]*(sent.count(word)/len(sent))
                    sent_vec += (vec * tf_idf)
                    weight_sum += tf_idf
            if weight_sum != 0:
                sent_vec /= weight_sum
            tfidf_sent_vectors.append(sent_vec)
        return tfidf_sent_vectors