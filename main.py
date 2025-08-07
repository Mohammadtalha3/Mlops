from finetunnig.Bert_finetunning import FineTunning
from DataPrepration.Data_prepration import DataPreprocessing
from utils.util  import stratify_split
from transformers import BertModel
import ray 

data_path= '/home/waheed/Desktop/Mlops/Data/dataset.csv'

# dataset=ray.data.read_csv(data_path)
# unique_vals=dataset.unique(column='tag')
# num_classes= len(unique_vals)


# llm= BertModel.from_pretrained('allenai/scibert_scivocab_uncased', return_dict= False)
Dp = DataPreprocessing(data_path)
data=Dp.data_reading()
train_ds, val_ds= stratify_split(data, label_col='tag', test_size=0.2)

print('This is the length of the training Data', len(train_ds))
# df=Dp.cleaned_data(data)    





# dim_embeddings=llm.config.hidden_size

# model=FineTunning(llm,dim_embeddings,p_dropout=0.5,num_classes= num_classes)


# print(model.named_parameters)

    
