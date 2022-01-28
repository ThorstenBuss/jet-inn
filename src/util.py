import sys
import json

import numpy as np
from sklearn import metrics
from torchvision import transforms

from JetEfpDataSet import JetEfpDataset
from preprocessing import *

def get_test_data(N, bkg_path, sig_path, skip_bkg=100000, skip_sig=0):
    test_data = JetEfpDataset(
        bkg_path,
        sig_path,
        start_bkg=skip_bkg,
        stop_bkg=skip_bkg+N,
        start_sig=skip_sig,
        stop_sig=skip_sig+N
    )

    return test_data[:]

def calc_roc(labels, measures, flip=False) :
    auc = metrics.roc_auc_score( labels, measures )
    if auc<0.5 and flip:
        auc = metrics.roc_auc_score( labels, -measures )
        fpr, tpr, _ = metrics.roc_curve( labels, -measures )
    else:
        fpr, tpr, _ = metrics.roc_curve( labels, measures )
    
    return fpr, tpr, auc

def get_perf_stats(labels, measures, flip=False, pos=0.5):
    fpr,tpr,auc = calc_roc (labels, measures, flip)
    idx = np.where(tpr > pos, tpr, np.inf).argmin()
    imtafe = 1/fpr[idx]
    return auc, imtafe

def loud_config():
    config_file = None
    if len(sys.argv)==2:
        config_file = sys.argv[1]
    with open('default.json') as json_file:
        extargs = json.load(json_file)
    if config_file:
        with open(config_file) as json_file:
            extargs.update(json.load(json_file))

    print(json.dumps(extargs, indent = 4))

    transfos = []
    for e in extargs['preprocessing']:
        transfo = globals()[e[0]]
        transfos.append(transfo(*e[1:]))

    extargs['preprocessing'] = transforms.Compose(transfos)

    return extargs
