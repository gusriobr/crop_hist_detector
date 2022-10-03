# from libc.math import exp as cexp
import tempfile

import matplotlib
import numpy as np
import pygraphviz
from matplotlib import pyplot as plt


def plot_model(model, precision=3, output_file=None, format = "png"):
    """
    Generamos hmm model plot using model internal values
    :param model:
    :param precision:
    :param output_file:
    :return:
    """
    G = pygraphviz.AGraph(directed=True)
    # out_edges = model.out_edge_count

    for state in model.states:
        if state.is_silent():
            color = 'grey'
        elif state.distribution.frozen:
            color = 'blue'
        else:
            color = 'red'

        G.add_node(state.name, color=color)

    transition_mtx = model.dense_transition_matrix()
    out_edges = list(model.graph.out_edges())
    for i in range(len(out_edges)):
        edge = out_edges[i]
        ini_state_id = get_state_id(edge[0].name, model.states)
        end_state_id = get_state_id(edge[1].name, model.states)
        p = transition_mtx[ini_state_id, end_state_id]

        G.add_edge(edge[0].name, edge[1].name, label=round(p, precision))

    if output_file is None:
        tmp_file = tempfile.NamedTemporaryFile()
        output_file = tmp_file.name

    G.draw(output_file, format=format, prog='dot')

def get_state_id(state_name, states):
    for i, st in enumerate(states):
        if st.name == state_name:
            return i
    raise Exception("state not found. " + str(state_name))
