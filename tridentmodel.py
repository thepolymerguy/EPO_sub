"""
TRIDENT MODEL IMPLEMENTATION

Date: 14 January 2023
Authors:  Egheosa Ogbomo & Amran Mohammed (The Polymer Guys)
Description: This script combines three ML-based models to identify whether an input text is related to green plastics or not.
    : Rename all /path/to/ -> correct path
"""

pip install transformers

########## IMPORTING REQUIRED PYTHON PACKAGES ##########
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
import math
import time
import csv
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import string

########## DEFINING FUNCTIONS FOR MODEL IMPLEMENTATIONS ##########

### Input data cleaner
all_stopwords = stopwords.words('english') # Making sure to only use English stopwords
extra_stopwords = ['ii', 'iii'] # Can add extra stopwords to be removed from dataset/input abstracts
all_stopwords.extend(extra_stopwords)
def clean_data(input, type='Dataframe'):
    """
    As preparation for use with the text similarity model, this function removes superfluous data from either a dataframe full of
    classifications, or an input string, in order for embeddings to be calculated for them. Removes:
    •	Entries with missing abstracts/descriptions/classifications/typos
    •	Duplicate entries
    •   Unnecessary punctuation
    •	Stop words (e.g., by, a , an, he, she, it)
    •  	URLs
    •	All entries are in the same language

    :param input: Either a dataframe or an individual string
    :param type: Tells fucntion whether input is a dataframe or an individual string
    :return: (if dataframe), returns a dataframe containing CPC classfication codes and their associated 'cleaned' description
    :return:  (if string), returns a 'cleaned' version of the input string
    """
    if type == 'Dataframe':
        cleaneddf = pd.DataFrame(columns=['Class', 'Description'])
        for i in range(0, len(input)):
            row_list = input.loc[i, :].values.flatten().tolist()
            noNaN_row = [x for x in row_list if str(x) != 'nan']
            listrow = []
            if len(noNaN_row) > 0:
                row = noNaN_row[:-1]
                row = [x.strip() for x in row]
                row = (" ").join(row)
                text_tokens = word_tokenize(row)  # splits abstracts into individual tokens to allow removal of stopwords by list comprehension
                Stopword_Filtered_List = [word for word in text_tokens if not word in all_stopwords]  # removes stopwords
                row = (" ").join(Stopword_Filtered_List)  # returns abstract to string form
                removechars = ['[', ']', '{', '}', ';', '(', ')', ',', '.', ':', '/', '-', '#', '?', '@', '£', '$']
                for char in removechars:
                    row = list(map(lambda x: x.replace(char, ''), row))

                row = ''.join(row)
                wnum = row.split(' ')
                wnum = [x.lower() for x in wnum]
                #remove duplicate words
                wnum = list(dict.fromkeys(wnum))
                #removing numbers
                wonum = []
                for x in wnum:
                    xv = list(x)
                    xv = [i.isnumeric() for i in xv]
                    if True in xv:
                        continue
                    else:
                        wonum.append(x)
                row = ' '.join(wonum)
                l = [noNaN_row[-1], row]
                cleaneddf.loc[len(cleaneddf)] = l
        cleaneddf = cleaneddf.drop_duplicates(subset=['Description'])
        cleaneddf.to_csv('/path/to/additionalcleanedclasses.csv', index=False)
        return cleaneddf

    elif type == 'String':
        text_tokens = word_tokenize(input)  # splits abstracts into individual tokens to allow removal of stopwords by list comprehension
        Stopword_Filtered_List = [word for word in text_tokens if not word in all_stopwords]  # removes stopwords
        row = (" ").join(Stopword_Filtered_List)  # returns abstract to string form
        removechars = ['[', ']', '{', '}', ';', '(', ')', ',', '.', ':', '/', '-', '#', '?', '@', '£', '$']
        for char in removechars:
            row = list(map(lambda x: x.replace(char, ''), row))
        row = ''.join(row)
        wnum = row.split(' ')
        wnum = [x.lower() for x in wnum]
        # remove duplicate words
        wnum = list(dict.fromkeys(wnum))
        # removing numbers
        wonum = []
        for x in wnum:
            xv = list(x)
            xv = [i.isnumeric() for i in xv]
            if True in xv:
                continue
            else:
                wonum.append(x)
        row = ' '.join(wonum)
        return row

### Mean Pooler
"""
Performs a mean pooling to reduce dimension of embedding
"""
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return tf.reduce_sum(token_embeddings * input_mask_expanded, 1) / tf.clip_by_value(input_mask_expanded.sum(1), clip_value_min=1e-9, clip_value_max=math.inf)

### Sentence Embedder
def sentence_embedder(sentences, model_path):
  """
  Calling the sentence similarity model to generate embeddings on input text.
  :param sentences: takes input text in the form of a string
  :param model_path: path to the text similarity model
  :return returns a (1, 384) embedding of the input text
  """
  tokenizer = AutoTokenizer.from_pretrained(model_path) #instantiating the sentence embedder using HuggingFace library
  model = AutoModel.from_pretrained(model_path, from_tf=True) #making a model instance
  encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
  # Compute token embeddings
  with torch.no_grad():
    model_output = model(**encoded_input)
  sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']) #outputs a (1, 384) tensor representation of input text
  return sentence_embeddings

### Sentence Embedding Preparation Function
def convert_saved_embeddings(embedding_string):
    """
    Preparing pre-computed embeddings for use for comparison with new abstract embeddings .
    Pre-computed embeddings are saved as tensors in string format so need to be converted back to numpy arrays in order to calculate cosine similarity.
    :param embedding_string:
    :return: Should be a single tensor with dims (,384) in string formate
    """
    embedding = embedding_string.replace('(', '')
    embedding = embedding.replace(')', '')
    embedding = embedding.replace('[', '')
    embedding = embedding.replace(']', '')
    embedding = embedding.replace('tensor', '')
    embedding = embedding.replace(' ', '')
    embedding = embedding.split(',')
    embedding = [float(x) for x in embedding]
    embedding = np.array(embedding)
    embedding = np.expand_dims(embedding, axis=0)
    embedding = torch.from_numpy(embedding)
    return embedding

### Generating Class Embeddings

Model_Path = '/path/to/Text Similarity Model Folder' ### Insert Path to MODEL DIRECTORY here
def class_embbedding_generator(classes):
    """
    This function is to be used to generate and save class embeddings
    Takes an input of 'cleaned' classes, generated by clean_data function, and computes vector representations of these classes (the embeddings) and saves them to csv
    :classes: Classes should be a dataframe including all of broad scope classes that are intended to be used to make comparisons with
    """
    class_embeddings = pd.DataFrame(columns=['Class', 'Description', 'Embedding'])
    for i in range(len(classes)):
        class_name = classes.iloc[i, 0]
        print(class_name)
        class_description = classes.iloc[i, 1]
        class_description_embedding = sentence_embedder(class_description, Model_Path)
        class_description_embedding = class_description_embedding.numpy()
        class_description_embedding = torch.from_numpy(class_description_embedding)
        embedding_entry = [class_name, class_description, class_description_embedding]
        class_embeddings.loc[len(class_embeddings)] = embedding_entry

### Broad Scope Classifier
Model_Path = '/path/to/Text Similarity Model Folder' ### Insert Path to MODEL DIRECTORY here
def broad_scope_class_predictor(class_embeddings, abstract_embedding, N=5, Sensitivity='Medium'):
    """
    Takes in pre-computed class embeddings and abstract texts, converts abstract text into
    :param class_embeddings: dataframe of class embeddings
    :param abstract: a single abstract embedding
    :param N: N highest matching classes to return, from highest to lowest, default is 5
    :return: predictions: a full dataframe of all the predictions on the 9500+ classes, HighestSimilarity: Dataframe of the N most similar classes
    """
    predictions = pd.DataFrame(columns=['Class Name', 'Score'])
    for i in range(len(class_embeddings)):
        class_name = class_embeddings.iloc[i, 0]
        embedding = class_embeddings.iloc[i, 2]
        embedding = convert_saved_embeddings(embedding)
        abstract_embedding = abstract_embedding.numpy()
        abstract_embedding = torch.from_numpy(abstract_embedding)
        cos = torch.nn.CosineSimilarity(dim=1)
        score = cos(abstract_embedding, embedding).numpy().tolist()
        result = [class_name, score[0]]
        predictions.loc[len(predictions)] = result
    greenpredictions = predictions.tail(52)
    if Sensitivity == 'High':
        Threshold = 0.5
    elif Sensitivity == 'Medium':
        Threshold = 0.40
    elif Sensitivity == 'Low':
        Threshold = 0.35
    GreenLikelihood = 'False'
    for i in range(len(greenpredictions)):
        score = greenpredictions.iloc[i, 1]
        if float(score) >= Threshold:
            GreenLikelihood = 'True'
            break
        else:
            continue
    HighestSimilarity = predictions.nlargest(N, ['Score'])
    print(HighestSimilarity)
    print(GreenLikelihood)
    return predictions, HighestSimilarity, GreenLikelihood

### Green Scope Classifier
Model_Path = '/path/to/Text Similarity Model Folder' ### Insert Path to MODEL DIRECTORY here
def green_scope_class_predictor(green_class_embeddings, abstract_embedding, N):
    """
    Takes in pre-computed green class embeddings and abstract texts, converts abstract text into embedding, calculates cosine similarity between abstract embeddings and each green class embedding,
    returns classes with highest cosine similarity
    :param class_embeddings: dataframe of class embeddings
    :param abstract: a single abstract embedding
    :param N: N highest matching classes to return, from highest to lowest, default is 5
    :return: predictions: a full dataframe of all the predictions on the selected number of Green Classes, HighestSimilarity: Dataframe of the N most similar classes
    """
    green_scope_predictions = pd.DataFrame(columns=['Class Name', 'Score'])
    for i in range(len(green_class_embeddings)):
        class_name = green_class_embeddings.iloc[i, 0]
        embedding = green_class_embeddings.iloc[i, 2]
        embedding = convert_saved_embeddings(embedding)
        abstract_embedding = abstract_embedding.numpy()
        abstract_embedding = torch.from_numpy(abstract_embedding)
        cos = torch.nn.CosineSimilarity(dim=1)
        score = cos(abstract_embedding, embedding).numpy().tolist()
        result = [class_name, score[0]]
        green_scope_predictions.loc[len(green_scope_predictions)] = result
    GreenHighestSimilarity = green_scope_predictions.nlargest(N, ['Score'])
    print(GreenHighestSimilarity)
    return green_scope_predictions, GreenHighestSimilarity

### Binary Scope Classifier
def predict_green(abstract_embedding, model):
    """
    Employing the Binary Scope classifier to determine likelihood that an input abstract is about Green Plastics
    :param abstract: Abstract in text form 
    :param model: 
    :return: gV
    """
    model_pred = model.predict([abstract])
    greenprob = model_pred[0]
    if greenprob <= 0.2:
        gV = 0
        #print('Patent is not about green plastics')
    elif greenprob > 0.2 and greenprob <= 0.5:
        gV = 0
        #print('Patent is likely not about green plastics')
    elif greenprob > 0.5 and greenprob <=0.75:
        gV = 0.5
        #print('Not clear from abstract whether or not patent is about green plastics')
    elif greenprob > 0.75 and greenprob <=0.95:
        gV = 1
       # print('Patent is likely about green plastics or either is about "green" or "plastics"')
    elif greenprob > 0.95:
        gV = 1
        #print('Patent very likely about is about green plastics')
    return gV

########## LOADING PRE-COMPUTED EMBEDDINGS ##########
class_embeddings = pd.read_csv('/path/to/MainClassEmbeddings.csv')
green_class_embeddings = pd.read_csv('path/to/GreenClassEmbeddings.csv')

abstract

########## Making Predictions ##########
Model = tf.keras.models.load_model('/path/to/densemodel folder') # Insert path to Dense model

abstract = """
Described herein are strength characteristics and biodegradation of articles produced using one or more “green” sustainable polymers and one or more carbohydrate-based polymers. A compatibilizer can optionally be included in the article. In some cases, the article can include a film, a bag, a bottle, a cap or lid therefore, a sheet, a box or other container, a plate, a cup, utensils, or the like.
"""
abstract= clean_data(abstract, type='String')
abstract_embedding = sentence_embedder(abstract, Model_Path)
Number = 10
broad_scope_predictions = broad_scope_class_predictor(class_embeddings, abstract_embedding, Number, Sensitivity='High')
BinaryScore = predict_green(abstract, Model)

if broad_scope_predictions[2] == 'True' and BinaryScore > 0.5:
  print('Input Text almost certainly related to a Green Plastic')
  green_scope_class_predictor(green_class_embeddings, abstract_embedding, Number)
elif broad_scope_predictions[2] == 'True' and BinaryScore == 0.5:
  print('Very Likely to be a Green Plastic')
  green_scope_class_predictor(green_class_embeddings, abstract_embedding, Number)
elif broad_scope_predictions[2] == 'False' and BinaryScore >=0.5:
  print('Not sure if input text is related to a Green Plastic or Not. Please Review Further')
  green_scope_class_predictor(green_class_embeddings, abstract_embedding, Number) 
elif broad_scope_predictions[2] == 'False' and BinaryScore <= 0.5:
  print('Input text  Very Unikely to be related to a Green Plastic')
else:
  print('Input text  Very Unikely to be related to a Green Plastic')
