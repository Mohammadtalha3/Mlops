from transformers import BertModel, BertTokenizer
import torch.functional as F
import torch.nn as nn
import numpy as np
import torch




class FineTunning(nn.Module):
    
    def __init__(self,llm, dim_embeddings,p_dropout, num_classes):
        super().__init__()
        self.llm= llm
        self.dropout= p_dropout
        self.fc1= torch.nn.Linear(dim_embeddings, num_classes)

    def forward(self, batch):
        ids,masks= batch['input_ids'], batch['attention_mask']
        seq,pool=self.llm(input_ids=ids, attention_mask= masks)
        z= self.dropout(pool)    
        z=self.fc1(z)
        return z
    
    
    @torch.inference_mode()
    def predict(self,batch):
        self.eval()
        z=self(batch)
        y_pred= torch.argmax(z,dim=1).cpu().numpy() # get predicted class index with highest prbability per sample
        return y_pred
    
    @torch.inference_mode()
    def predict_probability(self,batch):
        self.eval()
        z= self(batch)
        y_prob= F.softmax(z,dim=1).cpu().numpy()
        return y_prob



# if __name__=='__main__':
#     llm=BertModel.from_pretrained('allenai/scibert_scivocab_uncased', return_dict= False)
#     dim_embedding=llm.config.hidden_size
#     model= FineTunning(dim_embedding,p_dropout=0.5, num_classes=)
#     model.named_parameters()

   



    
