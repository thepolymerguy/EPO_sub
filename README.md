THE TRIDENT PATENT CLASSIFIER

A pseudo-ensemble model approach, featuring three separate machine learning algorithms working in tandem with each other to overcome the pitfalls of patent classification according to the CPC patent classification.

1.	We developed a Binary-Scope patent classifier (BiSC) that is be able to determine whether input text data should broadly be classified as a ‘Green Plastic’ patent or not. To implement this, we trained a classifier using a dataset split equally between patents/literature related to ‘Green Plastics’ (as we have defined below), and literature not related to ‘Green Plastics’. The output of this classification is a ‘True’ or ‘False’ prediction of whether the input text is a ‘Green Plastic’, as well as a score, between 0 and 1, that represents the confidence of this classifier.

3.	We developed a Broad-Scope patent classifier (BSC) trained on a dataset containing patent data and/or literature related to the 9500+ ‘Level 0’ classes that we have identified in the CPC classification scheme (from August 2022). This classifier can provide a score related to how likely input text to belongs to one of the 9500+ ‘Level 0’ classes (please see appendix for definition of Level 0 class), with a higher score representing a greater likelihood. This classifier outputs the N most likely classes that it believes the input text most likely belongs to.

3.	We developed a Green-Scope patent classifier (GSC) trained on a dataset containing patent data and/or literature related to the CPC classes that we have identified as being related to ‘Green Plastics’. The output of this classifier is the N classes that it believes the input literature most likely belongs to, where N is an optimal, user defined, number of classes.

**CONTENTS**

. The script to implement the Trident model, including the implementation of the BSC, BiSC and GSC, can be found in the ‘**TridentModel.py**’ file. 

The script used to extract the ‘Level-0’ class code and their full associated description can be found in the ‘ClassFinder.py’ file on our project’s GitHub. The script used to clean descriptions and generate the class code embeddings can be found in the ‘**MainClassEmbeddingsGenerator.py**’. The ‘Level-0’ class code embeddings can be found in the ‘**MainClassEmbeddings.csv**’ file.

The final list of ‘Green Plastic’ class code embeddings (approx. 100) can be found in the ‘**GreenClassEmbeddings.csv**’  document. The script used to conduct the first and second stage searches for ‘Green Plastic’ class codes can be found in the ‘**GreenClassFinder.py**’ file.

The scripts used to train and evaluate the BiSC can be found in the ‘**BiSC.py**’ (can also be found in ‘**BiSC.ipynb**’).

The files containing the weights and architecture for the simple ‘Dense’ model BiSC is contained in the ‘**densemodel**’ folder 

The code used to clean the dataset can be found in the ‘**DatasetCleaner.py**’ file.




