# -*- coding: utf-8 -*-
"""
Binary Scope Classifier

3 models tested:
    : Baseline model (TF-IDF sk-learn)
    : Simple Dense model
    : Transfer Learning model using Universal Sentence Encoder
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from google.colab import drive
drive.mount('/content/drive')

'''
Function to calculate accuracy, precision, recall and f1 (sklearn)
    : y_test -> test labels, y_pred -> predicted labels
'''
def cal_accuray(y_test, y_pred):
  acc = accuracy_score(y_test, y_pred) * 100
  pres, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

  res = {'accuracy':acc, 
         'precision': pres, 
         'recall': recall, 
         'f1': f1}

  return res

'''
Load data set from paths to greencsv and notgreencsv
'''
def loaddata(greencsv, notgreencsv):
  dfg = pd.read_csv(greencsv)
  dfn = pd.read_csv(notgreencsv)
  #shuffling dfn (as not all used in training dataset)
  dfn = dfn.sample(frac=1, random_state=25)
  #shuffling dfn
  data = pd.concat([dfg, dfn[0:len(dfg)]])
  data_shuff = data.sample(frac=1, random_state=25)
  
  return data_shuff

data_shuff = loaddata('/path/to/GreenPatents_Dataset.csv', '/path/to/NotGreenPatents_Dataset.csv')
data_shuff

train_sent, test_sent, train_lab, test_lab = train_test_split(data_shuff["Abstract"].to_numpy(),
                                                              data_shuff["GreenV"].to_numpy(),
                                                              test_size=0.1, # 10% of sample in test dataset
                                                              random_state=25)
#make sure type = string & as numpy array
train_sent = [str(x) for x in train_sent]
test_sent = [str(x) for x in test_sent]
train_sent = np.array(train_sent)
test_sent = np.array(test_sent)

#average length for word in sentence
def max_lengthval(X_train):
  avlength = round(sum([len(str(i).split()) for i in X_train])/len(X_train))
  return avlength

#evaluates model
def evaluate_model(model, X_test, y_test):
  model.evaluate(X_test, y_test)
  model_pred = tf.squeeze(tf.round(model.predict(X_test)))
  result = cal_accuray(y_test, model_pred)
  print(result)
  return result

'''
Baseline model:

Sk-learn baseline model, bayes
Using TF IDF to vectorise words 
'''
# create vectorisation and model
model_base = Pipeline([
    ("tfid", TfidfVectorizer()), # vectorizer
    ("clf", MultinomialNB()) # model
])

model_base.fit(train_sent, train_lab)

model_pred = model_base.predict(test_sent)
result_b = cal_accuray(test_lab, model_pred)
print(result_b)

'''
Model 1: simple dense model
Using Keras functional API (more flexible than sequential but can also use)
String (abstract) is tokenised and embedding vector is created.
'''
max_vocab_length = 10000 
# max number of words in our vocabulary - 10k words works well
# more vocabs and model performance decreases
max_length = max_lengthval(test_sent)

def create_densemodel(X_train, max_vocab_length, max_length):
  text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                      output_mode="int",
                                      output_sequence_length=max_length)

  text_vectorizer.adapt(train_sent)

  embedding = tf.keras.layers.Embedding(input_dim=max_vocab_length, # set input shape
                              output_dim=128, # set size of embedding vector # more info gets encoded
                              embeddings_initializer="uniform", # default, intialize randomly
                              input_length=max_length, 
                              name="embedding_1") 

  inputs = tf.keras.layers.Input(shape=(1,), dtype="string") # inputs are 1-dimensional strings
  x = text_vectorizer(inputs) # tokenise imputs
  x = embedding(x) # create embedding from tokenised inputs
  x = tf.keras.layers.GlobalAveragePooling1D()(x) # reduces dimension of embedding -> quicker computation
  outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x) # binary outputs, so sigmoid activation
  model = tf.keras.Model(inputs, outputs, name="model_dense") # make model
  return model

model_1 = create_densemodel(test_sent, max_vocab_length, max_length)

model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


model_1_history = model_1.fit(train_sent, 
                              train_lab,
                              epochs=5, #5 epochs works well to train model
                              validation_data=(test_sent, test_lab))

'''
Model 2: Transfer learning model using universal sentence encoder with pretrained weights
'''

USE = 'https://tfhub.dev/google/universal-sentence-encoder/4'
def create_transfermodel(USE):
  sentence_embedder_layer = hub.KerasLayer(USE, 
                                          input_shape = [],
                                          dtype = tf.string, 
                                          trainable = False, #keep pretrained weights
                                          name = 'USE')
  # build sequential model with sent embedder layer
  model = tf.keras.Sequential([
      sentence_embedder_layer, 
      tf.keras.layers.Dense(100, activation='relu'), 
      #100 works well, giving slight improvement over 64, more layers do not significantly improve the model
      tf.keras.layers.Dense(1, activation='sigmoid')], #binary so sigmoid output
      name = 'model')
  return model

model_2 = create_transfermodel(USE)
# compile model
model_2.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(), 
              metrics = ['accuracy'])
# fit model
model2_hist = model_2.fit(train_sent, train_lab, epochs=5,
                       validation_data=(test_sent, test_lab)
                       )
# again here 5 epochs enough for > 80% accuracy

"""
Same as file 'AbstractCleaner_gendataset.py' but just with single abtract inputs
    : If passing multiple abstracts can clean beforehand with 'AbstractCleaner_gendataset.py'
"""

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def clean_abst(sent):
  all_stopwords = stopwords.words('english') #Making sure to only use English stopwords
  extra_stopwords = ['ii', 'iii', 'the']
  all_stopwords.extend(extra_stopwords)

  sent = sent.split()
  sent = [x.strip() for x in sent]
  removechars = ['[', ']', '{', '}', ';', '(', ')', ',', '.', ':', '/', '-', '#', '?', '@', '£', '$']
  for char in removechars:
      sent = list(map(lambda x: x.replace(char, ''), sent))

  sent = [x.lower() for x in sent]
  sent = [word for word in sent if word not in all_stopwords]
  sent = list(dict.fromkeys(sent))
  wonum = []
  for x in sent:
      xv = list(x)
      xv = [i.isnumeric() for i in xv]
      if True in xv:
          continue
      else:
          wonum.append(x)
  cleansent = ' '.join(wonum)

  return cleansent

'''
To predict whether patent is about green plastics or not using input abstract
    : greenprob - value from the model.predict
'''
def predict_green(abst, model):
  abst = clean_abst(abst)
  model_pred = model.predict([abst])
  greenprob = model_pred[0]
  #print(greenprob)
  if greenprob <= 0.2:
    gV = 0
    print('Patent is not about green plastics')
  elif greenprob > 0.2 and greenprob <= 0.5:
    gV = 0
    print('Patent is likely not about green plastics')
  elif greenprob > 0.5 and greenprob <=0.75:
    gV = 0.5
    print('Not clear from abstract whether or not patent is about green plastics')
  elif greenprob > 0.75 and greenprob <=0.95:
    gV = 1
    print('Patent is likely about green plastics or either is about "green" or "plastics"')
  elif greenprob > 0.95:
    gV = 1
    print('Patent is about green plastics')
  return gV

result_1 = evaluate_model(model_1, test_sent, test_lab)

result_2 = evaluate_model(model_2, test_sent, test_lab)

all_results = pd.DataFrame({"Baseline": result_b, 
                            "Dense": result_1, 
                            "Transfer Learning": result_2})
all_results = all_results.transpose()    
all_results["accuracy"] = all_results["accuracy"]/100

'''
Plot to compare the performance of each of the three models
'''

import matplotlib
from matplotlib import pyplot as plt
all_results.plot(kind="bar", figsize=(6, 4), color=['#0081B4', '#EFA3C8', '#3C6255', '#FBC252'])
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.xticks(rotation=0, horizontalalignment="center")
plt.title("Model performace evaluation")
plt.ylim(0, 1)
plt.tight_layout()
#plt.savefig('/content/drive/MyDrive/binary_model/model_comp_eval.png', dpi=300, transparent=True)

#model_1.save("densemodel")
#model_2.save("transferlearning_model.h5")
# need to be loaded with custom_objects={"KerasLayer": hub.KerasLayer}
