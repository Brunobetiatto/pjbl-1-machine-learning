import pandas as pd 
import numpy as np 



def holdout_split(data, target=None, train_size=0.65, random_state=None): 


    if isinstance(data, (list, np.ndarray)): 
        data = np.array(data)
    elif isinstance(data, pd.DataFrame):
        data = data.values

    n = len(data) 
    np.random.seed(random_state) 
    indices = np.random.permutation(n) 

    train_end = int(train_size * n)
    train_idx, test_idx = indices[:train_end], indices[train_end:]

    if target is not None: 
        target = np.array(target)
        return data[train_idx], data[test_idx], target[train_idx], target[test_idx]
    else:
        return data[train_idx], data[test_idx] 
    

def holdout_indices(n, train_size=0.65, random_state=None): 
    """vai retornar (train_idx, test_idx)"""
    rng = np.random.RandomState(random_state) 
    indices = rng.permutation(n) 
    rng.shuffle(indices) 
    train_end = int(train_size * n) 

    return indices[:train_end], indices[train_end:]
    


