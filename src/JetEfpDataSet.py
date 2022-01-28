import numpy as np
import pandas as pd
import torch

class JetEfpDataset(torch.utils.data.Dataset):

    def __init__(self, bkg_path, sig_path=None,
            start_bkg=0, stop_bkg=None, start_sig=0, stop_sig=None,
            preprocessing=None, std=0.0):
        cols = ["efp_{0}".format(i+1) for i in range(8)]

        data_bkg = pd.read_hdf(
            bkg_path,
            key='table',
            start=start_bkg,
            stop=stop_bkg
        )[cols].to_numpy()

        labels_bkg = np.zeros( data_bkg.shape[0] )

        if sig_path:
            data_sig = pd.read_hdf(
                sig_path,
                key='table',
                start=start_sig,
                stop=stop_sig
            )[cols].to_numpy()

            labels_sig = np.ones( data_sig.shape[0] )
        
        if sig_path:
            data = np.concatenate((data_bkg, data_sig), axis=0)
            labels = np.concatenate((labels_bkg, labels_sig), axis=0)
        else:
            data = data_bkg
            labels = labels_bkg

        self.raw = data
    
        if preprocessing:
            data = preprocessing(data)

        self.data = torch.tensor(data.astype(np.float32))
        self.labels = torch.tensor(labels.astype(np.int64))
        self.preprocessing = preprocessing
        self.std = std

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.std > 0.0:
            if hasattr(idx, '__len__'):
                rand = torch.normal(0, self.std, size=(len(idx), self.data.shape[1]))
            else:
                rand = torch.normal(0, self.std, size=(self.data.shape[1],))
            return self.data[idx] + rand, self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]
