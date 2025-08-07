import ray
import re
import nltk
import re
import numpy as np
import pdb
from transformers import BertTokenizer
from nltk.corpus import stopwords
from transformers import BertModel, BertTokenizer

# pdb.set_trace()


class DataPreprocessing:
    def __init__(self, data_path):
        self.data_path= data_path
        nltk.download("stopwords")
        self.STOPWORDS= stopwords.words('english')
        self.indexed_classes={}
    
    def data_reading(self):
        dataset=ray.data.read_csv(self.data_path)
        # dataset.show(5)
        return dataset
    
    def Data_Cleaning(self,row):

        # target= data.unique(column='tag')
        # classes_to_index= {targ:index for index, targ in enumerate(target)}

        text = row['title'] + " " + row['description']
        text= text.lower()
        pattern = re.compile(r'\b(' + r"|".join(self.STOPWORDS) + r")\b\s*")
        text = pattern.sub('', text)
        text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)  # add spacing
        text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
        text = re.sub(" +", " ", text)  # remove multiple spaces
        text = text.strip()  # strip white space at the ends
        text = re.sub(r"http\S+", "", text)  #  remove links


        row['text']= text
        # row['classes_to_index']= classes_to_index

        return row
    
    def _generate_class_indexing(self,dataset):

        unique_tags= dataset.unique('tag')
        # tags= [row['tag'] for row in unique_tags]
        self.class_indexing= {tag: idx for idx,tag in enumerate(unique_tags)}

    
    def add_class_index(self, row):
        row['label']= self.class_indexing.get(row['tag'], -1)
        return row
    
    
    def tokenize(self,data):
        tokenizer= BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict= False)
        tokens=tokenizer(data['text'].tolist(), return_tensors= 'np',padding='longest')
        return dict(ids=tokens['input_ids'], masks=tokens['attention_mask'], targets= np.array(data['label']))
    

    def cleaned_data(self,dataset):
        dataset=dataset.map(self.Data_Cleaning)
        self._generate_class_indexing(dataset)
        dataset= dataset.map(self.add_class_index)
        output =dataset.map_batches(self.tokenize)
        # output = self.tokenize(dataset)
        return output
    
        

# data='/home/waheed/Desktop/Mlops/Data/dataset.csv'
# Dp=DataPreprocessing(data)
# data=Dp.data_reading()
# df=Dp.cleaned_data(data)
# df.show(10)

