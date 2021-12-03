import os
import importlib

import numpy as np
import torch


def convert_to_ndarray(arr):
    if torch.is_tensor(arr):
        arr = arr.numpy()
    if isinstance(arr, list):
        arr = np.array(arr)
    if not isinstance(arr, np.ndarray):
        raise Exception(f'Still no numpy ndarray! {type(arr)}')
    return arr


def load_as_dataloader(name, batch_size):
    mod = None
    for data in os.listdir(os.path.dirname(__file__)):
        if data == name:
            if os.path.exists(data):
                mod = importlib.import_module(data)
            elif os.path.exists(os.path.join('clflow', 'data', data)):
                mod = importlib.import_module(f'clflow.data.{data}')
            else:
                raise Exception(f'Could not import {data} from working dir {os.getcwd()}!')
            break
    train_ds, test_ds = mod.load()
    if batch_size == -1: # load complete data
        batch_size = len(train_ds)
        batch_size_test = len(test_ds)
    else:
        batch_size_test = batch_size
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader


def load_as_arrays(name):
    train_loader, test_loader = load_as_dataloader(name, batch_size=-1)
    X_train, y_train = next(iter(train_loader))
    X_test, y_test = next(iter(test_loader))

    X_train = convert_to_ndarray(X_train)
    X_test = convert_to_ndarray(X_test)
    y_train = convert_to_ndarray(y_train)
    y_test = convert_to_ndarray(y_test)

    return X_train, y_train, X_test, y_test
