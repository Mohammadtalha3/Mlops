from ray.air import Checkpoint, session
from ray.air.config import CheckpointConfig, DatasetConfig, RunConfig,ScalingConfig
from ray.train.torch import TorchCheckpoint, TorchTrainer
import ray.train as train
import torch.nn.functional as F
from ray.train.torch import get_device
import numpy as np
import torch


def pad_array(arr, dtype= np.int32):
    mac_len= max(len(row) for row in arr)
    padded_arr= np.zeros((arr.shape[0], mac_len), dtype=dtype)
    for i, row in enumerate(arr):
        padded_arr[i][:len(row)] = row
    return padded_arr


def collate_fn(batch):
    batch['ids']= pad_array(batch['ids'])
    batch['mask']= pad_array(batch['mask'])
    dtypes = {"ids": torch.int32, "masks": torch.int32, "targets": torch.int64}
    tensor_batch = {} 
    for key,array in  batch.items():
        tensor_batch[key] = torch.as_tensor(array, dtype=dtypes[key], device=get_device())
    return tensor_batch


def train_step(ds, batch_size,model, num_classes, loss_fn, optimizer):
    model.train()
    loss=0.0
    ds_generator= ds.iter_torch_batches(batch_size=batch_size,collate_fn= collate_fn)

    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad()
        z= model(batch)
        targets= F.one_hot(batch['targets'], num_classes=num_classes).float()
        J= loss_fn(z,targets)
        J.backward()
        optimizer.step()
        loss +=(J.detach().item()-loss)/(i+1)
    return loss





def eval_step(ds, batch_size, model, num_classes, loss_fn):
    """Eval step."""
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            z = model(batch)
            targets = F.one_hot(batch["targets"], num_classes=num_classes).float()  # one-hot (for loss_fn)
            J = loss_fn(z, targets).item()
            loss += (J - loss) / (i + 1)
            y_trues.extend(batch["targets"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)