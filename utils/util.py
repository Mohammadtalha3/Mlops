import numpy as np
# import  pandas as pd
import ray
import dataset as Dataset




def stratify_split(ds:Dataset, label_col= str, test_size= 0.2,seed:int= 42 ):

    unique_vals=ds.unique(column=label_col)
    train_items, val_items= [],[]
    rng = np.random.default_rng(seed=seed)

    for label_row in unique_vals:
        subset=ds.filter(lambda row: row[label_col]== label_row)

        count =subset.count()

        items= subset.take_all()

        rng.shuffle(items)

        split_idx= int(count * (1-test_size))

        train_items.extend(items[:split_idx])
        val_items.extend(items[split_idx:])

    
    train_ds=ray.data.from_items(train_items)
    val_ds=ray.data.from_items(val_items)

    return train_ds,val_ds










