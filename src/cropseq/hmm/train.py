import json
import operator
import pickle

import numpy as np
from pomegranate import DiscreteDistribution, HiddenMarkovModel
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

from cropseq import cfg

# sys.path.append("../../../")

cfg.configLog()

import logging


def train_model(X, n_states):
    logging.info("Training HMM from samples")
    model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=n_states, X=X,
                                           max_iterations=200, stop_threshold=0.1,
                                           n_jobs=4, verbose=True)
    logging.info("Model successfully trained!")
    return model


def cal_histogram(model, X):
    logging.info("Calculating Chi-square metric on histograms.")
    # calculate histograms
    h1 = np.histogram(X, bins=20)[0].astype(np.uint16)
    # generaten random sequences using the model
    sample = model.sample(X.shape[0], X.shape[1])
    sample = np.vstack(sample)
    h2 = np.histogram(sample, bins=20)[0].astype(np.float32)
    # chi square
    h2[h2 == 0] = 1e-5  # epsylon

    return np.sum(np.power(h1 - h2, 2) / h2)


def calc_likelihood(model, X):
    """
    given the sequences in the dataset X, calculate the sum of the log probability
    :param X:
    :return:
    """
    logging.info("Evaluating sequences likelihood")
    # iterate over the data instead of calling the full dataset to avoid problems with " x is not defined in a distribution
    lprob = 0
    n_tries = 0
    for i in range(0, X.shape[0]):
        try:
            lprob += model.summarize(X[i:i + 1, :])
            n_tries += 1
        except Exception as ex:
            # logging.exception(ex)
            pass
    logging.info("Number of valid sequences evaluated: {}".format(n_tries))
    return lprob


def calc_clf_metrics(model, X):
    """
    Using the last column of the dataset as y, try to predict the next value using n-1 sequences and compare to
    expected value using f1-score metric
    :param model:
    :param X:
    :return:
    """
    logging.info("Evaluating f1-score")
    y = X[:, -1]
    X_test = X[:, 0:-1]
    y_pred = []
    for i in range(0, len(X_test)):
        # run viterbi to get the mos probable path sequence and extract the observation distribution from the last state
        try:
            logp, path = model.viterbi(X_test[i])
            last_state = path[-1]
            distrib = last_state[1].distribution.parameters[0]
            max_key = max(distrib.items(), key=operator.itemgetter(1))[0]
        except:
            max_key = -1
        y_pred.append(max_key)

    return f1_score(y, y_pred, average="weighted"), cohen_kappa_score(y, y_pred, )


def eval_model(model, X_test):
    metrics = {}
    # calculate metrics
    metrics["chi"] = float(cal_histogram(model, X_test))
    metrics["lprob"] = float(calc_likelihood(model, X_test))
    f1, kappa = calc_clf_metrics(model, X_test)
    metrics["f1"] = float(f1)
    metrics["kappa"] = float(kappa)
    return metrics


def run_train_eval(X_train, X_test, number_states, num_trains=5):
    history = []
    num_model = 1
    for n_states in number_states:
        for n in range(0, num_trains):
            logging.info("_________________________________________")
            logging.info("[{}] Training model n_states {} n_iters = {}".format(num_model, n_states, n))
            model = train_model(X_train, n_states=n_states)
            metrics = eval_model(model, X_test)
            metrics["n_states"] = n_states
            history.append(metrics)
            logging.info("Metrics:\n {}".format(metrics))

            # store model
            with open(cfg.results("hmm_{}.pickle".format(num_model)), 'wb') as fmodel:
                pickle.dump(model, fmodel)
            # store metrics
            with open(cfg.results("history.json"), 'w') as fout:
                json_dumps_str = json.dumps(history, indent=4)
                print(json_dumps_str, file=fout)
            num_model += 1
            logging.info("_________________________________________")


if __name__ == '__main__':
    # use this to train models with sampled data
    # df = pickle.load(file=open(cfg.resource("dataset_sample.pickle"), 'rb'))
    # use this to train the model with full data
    df = pickle.load(file=open(cfg.resource("dataset.pickle"), 'rb'))

    # read sample dataset and extract train-test split
    cols_years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
    X = df[cols_years].values
    y = df["2021"].values  # just to call split method, we will discard this

    index = np.random.choice(X.shape[0], int(X.shape[0]*0.5), replace=False)
    X = X[index]
    y = y[index]

    # split using last column to stratify results
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    # dts_obj = (X_train, X_test, y_train, y_test)
    # pickle.dump(dts_obj, open(cfg.resource("test_train_split.pickle"), 'wb'))

    # load dataset
    X_train, X_test, y_train, y_test = pickle.load(open(cfg.resource("test_train_split.pickle"), 'rb'))

    run_train_eval(X_train, X_test, number_states=[12], num_trains=5)
