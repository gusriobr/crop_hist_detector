import sys, os
import pomegranate
import numpy as np
import random
import pandas as pd
import pickle

# sys.path.append("../../../")

from cropseq import cfg

import pandas as pd
import numpy as np
import pickle

from pomegranate import *
from cropseq import cfg

cfg.configLog()

import logging


def train_model(X):
    logging.info("Training HMM from samples")
    model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=4, X=X, max_iterations=1000, verbose=True)
    logging.info("Model successfully trained!")
    return model


if __name__ == '__main__':
    ds_path = cfg.resource('dataset.pickle')
    df = pd.read_pickle(ds_path)

    columns = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
    X = df[columns].values
    X = X[random.sample(range(0, X.shape[0]), 1000), :]  # sample 1000 randomly
    y = df['2021'].values

    # model = train_model(X)
    # pickle.dump(model, open(cfg.resource("hmm.pickle"), 'wb'))

    # Load trained modelo with 4 hidden states and discrite distribution for observation emission prob. distribution
    model = pickle.load(file=open(cfg.resource("hmm.pickle"), 'rb'))
    print(model)

    sample = X[0,:]
    print(sample)
