from ray.air import Checkpoint, session
from ray.air.config import CheckpointConfig, DatasetConfig, RunConfig, ScalingConfig
import ray.train as train
from ray.train.torch import TorchCheckpoint, TorchTrainer
import torch.nn.functional as F
from transformers import BertModel
from Bert_finetunning import  FineTunning
from torch import nn
import torch 
from  utils.distributed_system_utils import train_step, eval_step


train_loop_config = {
    "dropout_p": 0.5,
    "lr": 1e-4,
    "lr_factor": 0.8,
    "lr_patience": 3,
    "num_epochs": 10,
    "batch_size": 256,
    "num_classes": num_classes,
}

def train_loop(config):
    
    dropout_p = config["dropout_p"]
    lr = config["lr"]
    lr_factor = config["lr_factor"]
    lr_patience = config["lr_patience"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]


    train_ds=session.get_dataset_shard('train')
    val_ds=session.get_dataset_shard('val')
    

    llm= BertModel.from_pretrained("allenai/scibert_scivocab_uncased",return_dict= False)
    model=  FineTunning(llm,dim_embeddings=llm.config.hidden_size,p_dropout=dropout_p,num_classes=num_classes)
    model= train.torch.prepare_model(model)


    # Training Components

    loss_fn= nn.BCEWithLogitsLoss()
    optimizer= torch.optim.Adam(model.paramerters(), lr=lr)
    schedule= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode= 'min', factor= lr_factor,patience=lr_patience)


    # Training 

    bacth_size_per_worker=  batch_size// session.get_world_size()
    for epoch in range(num_epochs):
        
        train_loss= train_step(train_ds, bacth_size_per_worker,model, num_classes,optimizer)
        val_loss, _, _= eval_step(val_ds, bacth_size_per_worker,model, num_classes,loss_fn)
        schedule.step(val_loss)


        #checkpoints

        metrices= dict(epoch=epoch, lr= optimizer.param.group[0]['lr'],train_loss=train_loss, val_loss=val_loss )
        Checkpoint= TorchCheckpoint.from_model(model=model)
        session.report(metrices,Checkpoint=Checkpoint)




# Scaling config
scaling_config = ScalingConfig(
    num_workers=num_workers,
    use_gpu=bool(resources_per_worker["GPU"]),
    resources_per_worker=resources_per_worker,
    _max_cpu_fraction_per_node=0.8,
)

