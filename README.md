# Swiss dialect classification by Selena, Petra and Alena

This algorithm classifies sentences into the four Swiss dialects BE, BS, LU and ZH. It learns the predictions on a training file, takes the sentences from a test set and writes them with their predicted label to a result file.

Program call:
```
python classifier.py train.csv test.csv result.csv

train.csv:    Comma-separated CSV file with the columns "Text,Label"
test.csv:     Comma-separated CSV file with the columns "Id,Text"
results.csv:  Writes out comma-separated CSV file with the columns "Id,Prediction"
```


It runs in Python 3 and requires the following modules that need to be installed manually (python3 -m pip install [module]): 
```
pandas 
matplotlib
numpy
seaborn
sklearn
```
