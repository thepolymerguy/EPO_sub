
"""
Author: Egheosa Ogbomo
Date: 1st December 2022
Description:
Extracts the ‘Level-0’ class codes and their full associated descriptions from CPC class code CSVs. 
The script is used to clean descriptions and generate the class code embeddings.
"""

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
import math
import csv
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import string

############# Getting Class Descriptions ##############
"""
Gets Class Descriptions from all sections of the CPC and puts it into a dataframe. 
"""

Section = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']
Maintuples = []
for S in Section:
    #### Load in Datasets from the large text files
    with open(f'path/to/cpc CSVs/cpc-section-{S}_20220801.txt') as f:
        lines = f.readlines()

    for line in lines:
        Line_Elements = line.split("\t")
        ClassName = Line_Elements[0]
        Subclass_Level = 1
        classdescription = Line_Elements[2]
        classheader = ClassName[0]
        classline = [ClassName, Subclass_Level, classdescription, classheader]

        if ClassName.endswith('00') == True:
            Maintuples.append(classline)
        else:
            continue

metadata = []
descriptiondf = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

for Maintuple in Maintuples:
    descriptions = []
    with open(f'E:/Users/eeo21/Startup/CPC_Classifications_List/cpc-section-{Maintuple[3]}_20220801.txt') as f:
        lines = f.readlines()

    checked_lines = []
    for line in lines:
        Line_Elements = line.split("\t")
        ClassName = Line_Elements[0]
        Subclass_Level = Line_Elements[1]
        classdescription = Line_Elements[2]

        if ClassName != Maintuple[0]:
            classline = [ClassName, Subclass_Level, classdescription]
            checked_lines.append(classline)
            continue
        else:
            desiredclassline = (ClassName, Subclass_Level, classdescription)
            metadata.append(desiredclassline)
            descriptions.append(desiredclassline[0])
            descriptions.append(desiredclassline[2])
            checked_lines.reverse()
            LevelInitial = 0
            while LevelInitial > -1:
                for checked_line in checked_lines:
                    if len(checked_line[0]) == 4:
                        checked_line[1] = int(-1)
                    if len(checked_line[0]) == 3:
                        checked_line[1] = int(-2)
                    if len(checked_line[0]) == 1:
                        checked_line[1] = int(-3)
                    else:
                        checked_line[1] == int(0)
                    Subclass_Level = int(checked_line[1])
                    if LevelInitial - Subclass_Level == 1:
                        metadata.append(checked_line)
                        descriptions.append(checked_line[2])
                        LevelInitial -= 1
                    elif LevelInitial == -2:
                        break

    descriptions = descriptions[::-1]
    descriptiondf = descriptiondf.append(pd.Series(descriptions, index=descriptiondf.columns[:len(descriptions)]), ignore_index=True)

descriptiondf.to_csv('path/to/workingdir/MainClassDescriptions.csv', index=False) #Save full class descriptions to CSV

########## Generating Word Embeddings ##########

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return tf.reduce_sum(token_embeddings * input_mask_expanded, 1) / tf.clip_by_value(input_mask_expanded.sum(1), clip_value_min=1e-9, clip_value_max=math.inf)

all_stopwords = stopwords.words('english') #Making sure to only use English stopwords
extra_stopwords = ['ii', 'iii']
all_stopwords.extend(extra_stopwords)

def clean_data(input, type='Dataframe'):
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
        cleaneddf.to_csv('path/to/workingdir/Maincleanedclassesexpanded.csv', index=False) save cleaned classes
        
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

clean_data(classdf, type='Dataframe')
classes = pd.read_csv('path/to/workingdir/Maincleanedclassesexpanded.csv', index_col=False)

def sentence_embedder(sentences, model_path):
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModel.from_pretrained(model_path, from_tf=True)
  encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
  # Compute token embeddings
  with torch.no_grad():
    model_output = model(**encoded_input)
  sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
  return sentence_embeddings

# Generating Class Embeddings
class_embeddings = pd.DataFrame(columns=['Class', 'Description', 'Embedding'])
for i in range(len(classes)):
    class_name = classes.iloc[i, 0]
    class_description = classes.iloc[i, 1]
    class_description_embedding = sentence_embedder(class_description, 'path/to/text similarity model folder')
    embedding_entry = [class_name, class_description, class_description_embedding]
    class_embeddings.loc[len(class_embeddings)] = embedding_entry

class_embeddings.to_csv('path/to/workingdir/MainClassEmbeddings.csv') #Save Class Embeddings


