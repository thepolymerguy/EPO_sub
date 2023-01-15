import pandas as pd
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

'''
Clean dataset for green and not green abstracts
Dataset to be used for the Binary classification

Uncomment out #nltk.download('stopwords') and #nltk.download('punkt') to download stopwords

clean_data('file_to_be_cleaned.csv', 'cleaned_file.csv')
'''

def clean_data(fn,sn):
    
    all_stopwords = stopwords.words('english') #Making sure to only use English stopwords
    extra_stopwords = ['ii', 'iii', 'the']
    all_stopwords.extend(extra_stopwords)

    df = pd.read_csv(fn)
    print(df.head())
    heads = ['Abstract']
    dfclean = df
    for name in heads:
        for i in range(0, len(df[name])):
            print(i)
            if df[name][i] != 'NaN':
                sent = str(df[name][i])
                sent = sent.split()
                sent = [x.strip() for x in sent] # remove trailing chars
                removechars = ['[', ']', '{', '}', ';', '(', ')', ',', '.', ':', '/', '-', '#', '?', '@', '£', '$']
                for char in removechars:
                    # remove special chars
                    sent = list(map(lambda x: x.replace(char, ''), sent))

                sent = [x.lower() for x in sent]
                sent = [word for word in sent if word not in all_stopwords]
                sent = list(dict.fromkeys(sent))
                wonum = []
                for x in sent:
                    # remove all numbers
                    xv = list(x)
                    xv = [i.isnumeric() for i in xv]
                    if True in xv:
                        continue
                    else:
                        wonum.append(x)
                cleansent = ' '.join(wonum)
                dfclean[name][i] = cleansent
    
    
    dfclean = dfclean.drop_duplicates(subset=['Abstract'], keep='first')
    dfclean = dfclean.reset_index(drop=True)

    dfclean.to_csv(sn, index=False)

clean_data('filename.csv', 'savename.csv')
