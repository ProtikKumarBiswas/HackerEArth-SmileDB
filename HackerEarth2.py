# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:07:19 2019

@author: Protik
"""
import os
os.chdir('C:\\Users\\Protik\\Desktop\\HackerEarth')

import pandas as pd
import numpy as np
from sklearn.naive_bayes import ComplementNB
from time import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
import seaborn as sns
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from Vectorizers.AvgWord2Vec import AvgWord2Vec
from Vectorizers.TF_IDF_Word2Vec import TF_IDF_Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression

#-----------------------------------------------------------------------------------------------
#Extraction
data_train = pd.read_csv('dataset\\hm_train.csv')
data_test = pd.read_csv('dataset\\hm_test.csv')


out_classes = ['bonding', 'achievement', 'affection', 'leisure', 'enjoy_the_moment', 'nature', 'exercise']

#-----------------------------------------------------------------------------------------------
#Checking the integrity of the data
tmp = []
for idx in range(len(data_train)):
       if data_train[data_train.columns[4]][idx] in out_classes:
              continue
       tmp.append(idx)
       
data_train['hmid'].is_unique
data_test['hmid'].is_unique
'''
#-----------------------------------------------------------------------------------------------
#Pre-Processing
text_X = data_train[data_train.columns[2]].copy()

stopwords.words('english')

sstemmer = SnowballStemmer('english')
sstemmer.stem('With.'.lower())

#https://www.programiz.com/python-programming/examples/remove-punctuation'''
#def remove_punctuation(string):
#       punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
#       no_punct = ""
#       
#       for char in string:
#              if char not in punctuations:
#                     no_punct = no_punct + char                     
#       return(no_punct)
'''
def clean_text(text_X):
       cleaned_text = []
       for idx in tqdm(range(len(text_X))):
              words = text_X[idx].split()
              cleaned_words = []
              for w in words:
                     w.lower()
                     tmp_word = remove_punctuation(w)
                     if tmp_word in stopwords.words('english'):
                            continue
                     if len(tmp_word) <= 2:
                            continue
                     cleaned_words.append(sstemmer.stem(tmp_word))
              cleaned_text.append(' '.join(cleaned_words))
       
       return cleaned_text

text_len = []
for idx in range(len(cleaned_text)):
       text_len.append(len(cleaned_text[idx]))
       
np.max(text_len)

plt.plot(range(len(text_X)), text_len)

data_train[data_train.columns[4]].value_counts()
'''
#-----------------------------------------------------------------------------------------------
#train test split
X = data_train[data_train.columns[1:4]]
y = data_train[data_train.columns[4]]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .8)

#-----------------------------------------------------------------------------------------------
#Vectoriztion
'''
avgw2v = AvgWord2Vec()
X = avgw2v.fit(x = text_X, vector_size = 500)
'''
def vectorize(X_train, X_test, vectorizer_ = 'TF_IDF'):
       
       vectorizer = 0
       if vectorizer_ == 'TF_IDF':
              print('TF_IDF Running ....\n')
              vectorizer = TfidfVectorizer(min_df = 10, max_features = 5000)
              X_temp = pd.DataFrame(vectorizer.fit_transform(X_train[X_train.columns[1]]).toarray(), index = X_train.index)
              X_train = X_train.join(X_temp)
              X_temp = pd.DataFrame(vectorizer.transform(X_test[X_test.columns[1]]).toarray(), index = X_test.index)
              X_test = X_test.join(X_temp)
       elif vectorizer_ == 'AvgWord2Vec':
              vectorizer = AvgWord2Vec()
              X_train = vectorizer.fit(X_train, vector_size = 5000)
              X_test = vectorizer.transform(X_test)
       elif vectorizer_ == 'TF_IDF_Word2Vec':
              vectorizer = TF_IDF_Word2Vec()
              X_train = vectorizer.fit(X_train, vector_size = 5000)
              X_test = vectorizer.transform(X_test)
       else:
              return vectorizer_
       
       if vectorizer_ != 'TF_IDF':
              X_temp = pd.DataFrame(X_train[X_train.columns[1]], index = X_train.index)
              X_train = X_train.join(X_temp)
              X_temp = pd.DataFrame(X_test[X_test.columns[1]], index = X_test.index)
              X_test = X_test.join(X_temp)
              
       X_train = X_train.drop(columns = X_train.columns[1])
       X_test = X_test.drop(columns = X_test.columns[1])
       
       print('Replacement Running ....\n')
       X_train[X_train.columns[0]] = X_train[X_train.columns[0]].replace('24h', 0)
       X_train[X_train.columns[0]] = X_train[X_train.columns[0]].replace('3m', 1)
       X_train = X_train.astype(float)
       
       X_test[X_test.columns[0]] = X_test[X_test.columns[0]].replace('24h', 0)
       X_test[X_test.columns[0]] = X_test[X_test.columns[0]].replace('3m', 1)
       X_test = X_test.astype(float)
       
       print('Converting to csr_matrix Running ....\n')
       X_train = csr_matrix(X_train.values)
       X_test = csr_matrix(X_test.values)
       
       print("the shape of", vectorizer_," train vector ", X_train.shape)
       print("the shape of ", vectorizer_,"test vector ", X_test.shape)
       
       return X_train, X_test, vectorizer

X_train, X_test, tfidf_vec = vectorize(X_train, X_test, 'TF_IDF')


#-----------------------------------------------------------------------------------------------
#Hyperparameter tuning

CompNB = ComplementNB()


check = time()
Preff_Alpha = [10 ** x for x in range(-7,8)]
para_dic = {'alpha' : Preff_Alpha}

clf = GridSearchCV(CompNB, para_dic, cv=4, scoring='f1_weighted', n_jobs = 12, return_train_score = True)
clf.fit(X_train, y_train)

train_f1= clf.cv_results_['mean_train_score']
train_f1_std= clf.cv_results_['std_train_score']
cv_f1 = clf.cv_results_['mean_test_score'] 
cv_f1_std= clf.cv_results_['std_test_score']

print('Grid Search Time : ', time() - check)

results = clf.cv_results_
#-----------------------------------------------------------------------------------------------
#Ploting to choose the best Hyperparameter
Alpha = list(clf.cv_results_['param_alpha'])
x_axis = np.log10(Alpha)


plt.plot(np.sort(x_axis), train_f1[np.argsort(x_axis)], label='Train f1 Score')
#https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(np.sort(x_axis),train_f1[np.argsort(x_axis)] - train_f1_std[np.argsort(x_axis)],train_f1[np.argsort(x_axis)] + train_f1_std[np.argsort(x_axis)],alpha=0.2,color='darkblue')

plt.plot(np.sort(x_axis), cv_f1[np.argsort(x_axis)], label='CV f1 Score')
#https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(np.sort(x_axis),cv_f1[np.argsort(x_axis)] - cv_f1_std[np.argsort(x_axis)],cv_f1[np.argsort(x_axis)] + cv_f1_std[np.argsort(x_axis)],alpha=0.2,color='darkorange')

plt.legend()
plt.xlabel("Hyperparameter")
plt.ylabel("f1 Score")
plt.title("ERROR PLOT")
plt.show()

best_Alpla = 1
#-----------------------------------------------------------------------------------------------
#Training

check = time()

new_CompNB = ComplementNB(alpha = best_Alpla)
new_CompNB.fit(X_train, y_train)

y_pred = new_CompNB.predict(X_test)
y_train_pred = new_CompNB.predict(X_train)

print('Train Time + Prediction Time( for both y_pred and y_train_pred ) : ', time() - check)

#-----------------------------------------------------------------------------------------------
#Test Results


print("Train confusion matrix")
con_matrix  = pd.DataFrame(confusion_matrix(y_train, y_train_pred), index = ['Actual'+str(u) for u in range(7)], columns = ['Predicted'+str(u) for u in range(7)])
sns.heatmap(con_matrix, annot=True, fmt="d")
plt.show()

print("Test confusion matrix")
con_matrix  = pd.DataFrame(confusion_matrix(y_test, y_pred), index = ['Actual'+str(u) for u in range(7)], columns = ['Predicted'+str(u) for u in range(7)])
sns.heatmap(con_matrix, annot=True, fmt="d")
plt.show()

print(f1_score(y_train, y_train_pred, average = 'weighted'), accuracy_score(y_train, y_train_pred))
print(f1_score(y_test, y_pred, average = 'weighted'), accuracy_score(y_test, y_pred))


#-----------------------------------------------------------------------------------------------
#Acctual Prediction

CompNB = ComplementNB(alpha = best_Alpla)

X_vec, X_test_,tfidf_vec = vectorize(X, data_test[data_test.columns[1:4]])


check = time()
CompNB.fit(X_vec, y)
print('Train Time: ', time() - check)



check = time()
y_pred_ = CompNB.predict(X_test_)
print('Prediction: ', time() - check)

#-----------------------------------------------------------------------------------------------
#Final submission
Final = []
for idx in range(len(y_pred_)):
       Final.append((data_test[data_test.columns[0]][idx], y_pred_[idx]))
Final_Df = pd.DataFrame(Final, columns = ('hmid', 'predicted_category'))

Final_Df.to_csv('submission5.csv', index = False)

#-----------------------------------------------------------------------------------------------
#Important Features
CompNB.classes_
for i in range(len(CompNB.classes_)):
       pos_class_prob_sorted = CompNB.feature_log_prob_[i, :].argsort()

       print("Top 10 most important features for the category", CompNB.classes_[i], ':\n')
       print(np.take(tfidf_vec.get_feature_names(), pos_class_prob_sorted[-10:]))
