import logging
import pickle

from matplotlib import pyplot as plt

from cropseq import cfg
# sys.path.append("../../../")
from cropseq.graph import plot_model
from cropseq.hmm.train import calc_clf_metrics

cfg.configLog()

model = pickle.load(file=open(cfg.results("hmm_granade.pickle"), 'rb'))
df = pickle.load(file=open(cfg.resource("dataset.pickle"), 'rb'))

cols_years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
X = df[cols_years].values
# y = df["2021"].values  # just to call split method, we will discard this

# lets run the model against the full dataset and check the f1 metrics and test some samples, compared with the inital trie
metrics = calc_clf_metrics(model, X)
result = {"f1": metrics[0], "kappa": metrics[1]}
logging.info("Classification metrics: {}".format(result))


# plot_model(model, format="svg ", output_file="/tmp/salida.svg")
