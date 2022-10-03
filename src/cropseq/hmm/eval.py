import json
import operator
import pickle
from matplotlib import pyplot as  plt

import numpy as np
from pomegranate import DiscreteDistribution, HiddenMarkovModel
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

from cropseq import cfg

# sys.path.append("../../../")
from cropseq.graph import plot_model

cfg.configLog()

model = pickle.load(file=open(cfg.results("hmm_granade.pickle"), 'rb'))

print("Número de estados: {}".format(len(model.states)))
model.plot(precision = 3)

import tempfile
print(str(tempfile.tempdir))

plot_model(model, format="svg ", output_file="/tmp/salida.svg")
plt.figure(figsize=(30, 20), dpi=120)
plt.show()



pass

