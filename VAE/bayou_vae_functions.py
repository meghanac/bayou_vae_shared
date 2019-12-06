from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf

import argparse
import os
import sys
import simplejson as json
import textwrap
import pickle
from collections import Counter
from copy import deepcopy
import ijson
import re
from tensorflow.python.client import device_lib
import time
import random
from itertools import chain
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from graphviz import Digraph
import graphviz
from tensorflow.contrib import legacy_seq2seq as seq2seq
from sklearn.manifold import TSNE

from bayou_vae_utils import ModelSettings

# Utils


CONFIG_GENERAL = ['model', 'latent_size', 'batch_size', 'num_epochs',
                  'learning_rate', 'print_step', 'checkpoint_step']
CONFIG_ENCODER = ['name', 'units', 'num_layers', 'tile', 'max_depth', 'max_nums', 'ev_drop_prob', 'ev_call_drop_prob']
CONFIG_DECODER = ['units', 'num_layers', 'max_ast_depth']
CONFIG_REVERSE_ENCODER = ['units', 'num_layers', 'max_ast_depth']
CONFIG_INFER = ['vocab', 'vocab_size']


def get_var_list():
    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')
    rev_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Reverse_Encoder')
    bayou_vars = decoder_vars + rev_encoder_vars
    var_dict = {'all_vars': all_vars,
                'decoder_vars': decoder_vars,
                'bayou_vars': bayou_vars,
                'rev_encoder_vars': rev_encoder_vars}
    return var_dict


def plot_probs(prob_vals, fig_name="rankedProb.pdf", logx=False):
    plt.figure()
    plot_path = os.path.join(os.getcwd(), 'generation')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    plt.grid()
    plt.title("Probability With Ranks")
    if logx:
        plt.semilogx(prob_vals)
    else:
        plt.plot(prob_vals)
    plt.xlabel("Ranks->")
    plt.ylabel("Log Probabilities")
    plt.savefig(os.path.join(plot_path, fig_name), bbox_inches='tight')
    # plt.show()
    return


def length(tensor):
    elems = tf.sign(tf.reduce_max(tensor, axis=2))
    return tf.reduce_sum(elems, axis=1)


# split s based on camel case and lower everything (uses '#' for split)
def split_camel(s):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1#\2', s)  # UC followed by LC
    s1 = re.sub('([a-z0-9])([A-Z])', r'\1#\2', s1)  # LC followed by UC
    split = s1.split('#')
    return [s.lower() for s in split]


def normalize_log_probs(probs):
    sum = -1 * np.inf
    for prob in probs:
        sum = np.logaddexp(sum, prob)

    for i in range(len(probs)):
        probs[i] -= sum
    return probs


def rank_statistic(_rank, total, prev_hits, cutoff):
    cutoff = np.array(cutoff)
    hits = prev_hits + (_rank < cutoff)
    prctg = hits / total
    return hits, prctg


# convert JSON to config
def read_config(js):
    config = argparse.Namespace()

    for attr in CONFIG_GENERAL:
        config.__setattr__(attr, js[attr])

    config.decoder = argparse.Namespace()
    config.reverse_encoder = argparse.Namespace()

    for attr in CONFIG_DECODER:
        config.decoder.__setattr__(attr, js['decoder'][attr])

    for attr in CONFIG_INFER:
        config.decoder.__setattr__(attr, js['decoder'][attr])
    chars_dict = dict()
    for item, value in config.decoder.vocab.items():
        chars_dict[value] = item
    config.decoder.__setattr__('chars', chars_dict)

    for attr in CONFIG_REVERSE_ENCODER:
        config.reverse_encoder.__setattr__(attr, js['reverse_encoder'][attr])
    for attr in CONFIG_INFER:
        config.reverse_encoder.__setattr__(attr, js['reverse_encoder'][attr])
    return config

# convert config to JSON
def dump_config(config):
    js = {}

    for attr in CONFIG_GENERAL:
        js[attr] = config.__getattribute__(attr)

    js['decoder'] = {attr: config.decoder.__getattribute__(attr) for attr in CONFIG_DECODER + CONFIG_INFER}
    # added code for reverse encoder
    js['reverse_encoder'] = {attr: config.reverse_encoder.__getattribute__(attr) for attr in
                             CONFIG_REVERSE_ENCODER + CONFIG_INFER}
    return js


def gather_calls(node):
    """
    Gathers all call nodes (recursively) in a given AST node

    :param node: the node to gather calls from
    :return: list of call nodes
    """

    if type(node) is list:
        return list(chain.from_iterable([gather_calls(n) for n in node]))
    node_type = node['node']
    if node_type == 'DSubTree':
        return gather_calls(node['_nodes'])
    elif node_type == 'DBranch':
        return gather_calls(node['_cond']) + gather_calls(node['_then']) + gather_calls(node['_else'])
    elif node_type == 'DExcept':
        return gather_calls(node['_try']) + gather_calls(node['_catch'])
    elif node_type == 'DLoop':
        return gather_calls(node['_cond']) + gather_calls(node['_body'])
    else:  # this node itself is a call
        return [node]


# Node


CHILD_EDGE = True
SIBLING_EDGE = False

MAX_LOOP_NUM = 3
MAX_BRANCHING_NUM = 3


class TooLongLoopingException(Exception):
    pass


class TooLongBranchingException(Exception):
    pass


class Node():
    def __init__(self, call, child=None, sibling=None):
        self.val = call
        self.child = child
        self.sibling = sibling

    def addAndProgressSiblingNode(self, predictionNode):
        self.sibling = predictionNode
        return self.sibling

    def addAndProgressChildNode(self, predictionNode):
        self.child = predictionNode
        return self.child

    def check_nested_branch(self):
        head = self
        count = 0
        while (head != None):
            if head.val == 'DBranch':
                count_Else = head.child.child.check_nested_branch()  # else
                count_Then = head.child.sibling.check_nested_branch()  # then
                count = 1 + max(count_Then, count_Else)
                if count > MAX_BRANCHING_NUM:
                    raise TooLongBranchingException
            head = head.sibling
        return count

    def check_nested_loop(self):
        head = self
        count = 0
        while (head != None):
            if head.val == 'DLoop':
                count = 1 + head.child.child.check_nested_loop()

                if count > MAX_LOOP_NUM:
                    raise TooLongLoopingException
            head = head.sibling
        return count

    def depth_first_search(self):

        buffer = []
        stack = []
        dfs_id = None
        parent_id = 0
        if self is not None:
            stack.append((self, parent_id, SIBLING_EDGE))
            dfs_id = 0

        while (len(stack) > 0):

            item_triple = stack.pop()
            item = item_triple[0]
            parent_id = item_triple[1]
            edge_type = item_triple[2]

            buffer.append((item.val, parent_id, edge_type))

            if item.sibling is not None:
                stack.append((item.sibling, dfs_id, SIBLING_EDGE))

            if item.child is not None:
                stack.append((item.child, dfs_id, CHILD_EDGE))

            dfs_id += 1

        return buffer

    def iterateHTillEnd(self):
        head = self
        while (head.sibling != None):
            head = head.sibling
        return head


def get_ast_from_json(js):
    ast = get_ast(js, idx=0)
    real_head = Node("DSubTree")
    real_head.sibling = ast
    return real_head


def get_ast(js, idx=0):
    # print (idx)
    cons_calls = []
    i = idx
    curr_Node = Node("Dummy_Fist_Sibling")
    head = curr_Node
    while i < len(js):
        if js[i]['node'] == 'DAPICall':
            curr_Node.sibling = Node(js[i]['_call'])
            curr_Node = curr_Node.sibling
        else:
            break
        i += 1
    if i == len(js):
        curr_Node.sibling = Node('STOP')
        curr_Node = curr_Node.sibling
        return head.sibling

    node_type = js[i]['node']

    if node_type == 'DBranch':
        nodeC = read_DBranch(js[i])

        future = get_ast(js, i + 1)
        branching = Node('DBranch', child=nodeC, sibling=future)

        curr_Node.sibling = branching
        curr_Node = curr_Node.sibling
        return head.sibling

    if node_type == 'DExcept':
        nodeT = read_DExcept(js[i])

        future = get_ast(js, i + 1)

        exception = Node('DExcept', child=nodeT, sibling=future)
        curr_Node.sibling = exception
        curr_Node = curr_Node.sibling
        return head.sibling

    if node_type == 'DLoop':
        nodeC = read_DLoop(js[i])
        future = get_ast(js, i + 1)

        loop = Node('DLoop', child=nodeC, sibling=future)
        curr_Node.sibling = loop
        curr_Node = curr_Node.sibling

        return head.sibling


def read_DLoop(js_branch):
    # assert len(pC) <= 1
    nodeC = get_ast(js_branch['_cond'])  # will have at most 1 "path"
    nodeB = get_ast(js_branch['_body'])
    nodeC.child = nodeB

    return nodeC


def read_DExcept(js_branch):
    nodeT = get_ast(js_branch['_try'])
    nodeC = get_ast(js_branch['_catch'])
    nodeC.child = nodeT

    return nodeC


def read_DBranch(js_branch):
    nodeC = get_ast(js_branch['_cond'])  # will have at most 1 "path"
    # assert len(pC) <= 1
    nodeT = get_ast(js_branch['_then'])
    # nodeC.child = nodeT
    nodeE = get_ast(js_branch['_else'])

    nodeC.sibling = nodeE
    nodeC.child = nodeT

    return nodeC


def colnum_string(n):
    n = n + 26 * 26 * 26 + 26 * 26 + 26 + 1
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string


def plot_path(model_settings, i, path, prob):
    dot = Digraph(comment='Program AST', format='eps')
    dot.node(str(prob), str(prob)[:6])
    for dfs_id, item in enumerate(path):
        node_value, parent_id, edge_type = item
        dot.node(str(dfs_id), node_value)
        label = 'child' if edge_type else 'sibling'
        label += " / " + str(dfs_id)
        if dfs_id > 0:
            dot.edge(str(parent_id), str(dfs_id), label=label, constraint='true', direction='LR')

    stri = colnum_string(i)
    dot.render(model_settings.output_plots_path + 'program-ast-' + stri + '-' + '.gv')
    return dot


# Data Reader


class Reader:
    def __init__(self, model_settings, config, infer=False):
        self.infer = infer
        self.config = config

        random.seed(12)
        # read the raw targets
        print('Reading data file...')
        raw_data_points = self.read_data(model_settings.data_path, infer)

        config.num_batches = int(len(raw_data_points) / config.batch_size)
        assert config.num_batches > 0, 'Not enough data'

        sz = config.num_batches * config.batch_size
        raw_data_points = raw_data_points[:sz]

        # wrangle the evidences and targets into numpy arrays
        self.nodes = np.zeros((sz, config.decoder.max_ast_depth), dtype=np.int32)
        self.edges = np.zeros((sz, config.decoder.max_ast_depth), dtype=np.bool)
        self.targets = np.zeros((sz, config.decoder.max_ast_depth), dtype=np.int32)

        for i, path in enumerate(raw_data_points):
            len_path = min(len(path), config.decoder.max_ast_depth)
            mod_path = path[:len_path]

            self.nodes[i, :len_path] = [p[0] for p in mod_path]
            self.edges[i, :len_path] = [p[1] for p in mod_path]
            self.targets[i, :len_path] = [p[2] for p in mod_path]

        if model_settings.save_outputs:
            os.makedirs(model_settings.output_data_path, exist_ok=True)
            with open(model_settings.output_data_path + 'nodes_edges_targets.txt', 'wb') as f:
                pickle.dump([self.nodes, self.edges, self.targets], f)

            jsconfig = dump_config(config)
        # with open(os.path.join(model_settings.output_config_path, 'config.json'), 'w') as f:
        #     json.dump(jsconfig, fp=f, indent=2)

            with open((model_settings.output_data_path + 'config.json'), 'w') as f:
                json.dump(jsconfig, fp=f, indent=2)

    def read_data(self, filename, infer):
        data_points = []
        done, ignored_for_branch, ignored_for_loop = 0, 0, 0
        self.decoder_api_dict = decoderDict(infer, self.config.decoder)

        f = open(filename, 'rb')
        for program in ijson.items(f, 'programs.item'):
            try:
                ast_node_graph = get_ast_from_json(program['ast']['_nodes'])

                ast_node_graph.sibling.check_nested_branch()
                ast_node_graph.sibling.check_nested_loop()

                path = ast_node_graph.depth_first_search()
                parsed_data_array = []
                for i, (curr_node_val, parent_node_id, edge_type) in enumerate(path):
                    curr_node_id = self.decoder_api_dict.get_or_add_node_val_from_callMap(curr_node_val)
                    # now parent id is already evaluated since this is top-down breadth_first_search
                    parent_call = path[parent_node_id][0]
                    parent_call_id = self.decoder_api_dict.get_node_val_from_callMap(parent_call)

                    if i > 0 and not (
                            # I = 0 denotes DSubtree ----sibling---> DSubTree
                            curr_node_id is None or parent_call_id is None):
                        parsed_data_array.append((parent_call_id, edge_type, curr_node_id))

                data_points.append(parsed_data_array)
                done += 1

            except TooLongLoopingException as e1:
                ignored_for_loop += 1

            except TooLongBranchingException as e2:
                ignored_for_branch += 1

            if done % 100000 == 0:
                print('Extracted data for {} programs'.format(done), end='\n')
                # break

        print('{:8d} programs/asts in training data'.format(done))
        print('{:8d} programs/asts missed in training data for loop'.format(ignored_for_loop))
        print('{:8d} programs/asts missed in training data for branch'.format(ignored_for_branch))

        # randomly shuffle to avoid bias towards initial data points during training
        random.shuffle(data_points)

        return data_points


def read_input_program(model_settings, config, filename):
    data_points = []
    done, ignored_for_branch, ignored_for_loop = 0, 0, 0
    infer = False
    save = None
    decoder_api_dict = decoderDict(infer, config.decoder)

    f = open(filename, 'rb')
    for program in ijson.items(f, 'programs.item'):
        try:
            ast_node_graph = get_ast_from_json(program['ast']['_nodes'])

            ast_node_graph.sibling.check_nested_branch()
            ast_node_graph.sibling.check_nested_loop()

            path = ast_node_graph.depth_first_search()
            parsed_data_array = []
            for i, (curr_node_val, parent_node_id, edge_type) in enumerate(path):
                curr_node_id = decoder_api_dict.get_or_add_node_val_from_callMap(curr_node_val)
                # now parent id is already evaluated since this is top-down breadth_first_search
                parent_call = path[parent_node_id][0]
                parent_call_id = decoder_api_dict.get_node_val_from_callMap(parent_call)

                if i > 0 and not (
                        curr_node_id is None or parent_call_id is None):  # I = 0 denotes DSubtree ----sibling---> DSubTree
                    parsed_data_array.append((parent_call_id, edge_type, curr_node_id))

            data_points.append(parsed_data_array)
            done += 1

        except (TooLongLoopingException) as e1:
            ignored_for_loop += 1

        except (TooLongBranchingException) as e2:
            ignored_for_branch += 1

        if done % 100000 == 0:
            print('Extracted data for {} programs'.format(done), end='\n')
            # break

    # randomly shuffle to avoid bias towards initial data points during training
    random.shuffle(data_points)
    raw_data_points = data_points

    config.num_batches = int(len(raw_data_points) / config.batch_size)
    assert config.num_batches > 0, 'Not enough data'

    sz = config.num_batches * config.batch_size
    raw_data_points = raw_data_points[:sz]

    # setup input and target vocab
    config.decoder.vocab, config.decoder.vocab_size = decoder_api_dict.get_call_dict()
    config.reverse_encoder.vocab, config.reverse_encoder.vocab_size = decoder_api_dict.get_call_dict()

    # wrangle the evidences and targets into numpy arrays
    nodes = np.zeros((sz, config.decoder.max_ast_depth), dtype=np.int32)
    edges = np.zeros((sz, config.decoder.max_ast_depth), dtype=np.bool)

    for i, path in enumerate(raw_data_points):
        len_path = min(len(path), config.decoder.max_ast_depth)
        mod_path = path[:len_path]

        nodes[i, :len_path] = [p[0] for p in mod_path]
        edges[i, :len_path] = [p[1] for p in mod_path]

    return np.array(nodes), np.array(edges)


def read_program_from_json(config, program):
    data_points = []
    done, ignored_for_branch, ignored_for_loop = 0, 0, 0
    infer = True
    decoder_api_dict = decoderDict(infer, config.decoder)


    try:
        ast_node_graph = get_ast_from_json(program['ast']['_nodes'])

        ast_node_graph.sibling.check_nested_branch()
        ast_node_graph.sibling.check_nested_loop()

        path = ast_node_graph.depth_first_search()
        parsed_data_array = []
        for i, (curr_node_val, parent_node_id, edge_type) in enumerate(path):
            curr_node_id = decoder_api_dict.get_or_add_node_val_from_callMap(curr_node_val)
            # now parent id is already evaluated since this is top-down breadth_first_search
            parent_call = path[parent_node_id][0]
            parent_call_id = decoder_api_dict.get_node_val_from_callMap(parent_call)

            if i > 0 and not (
                    curr_node_id is None or parent_call_id is None):  # I = 0 denotes DSubtree ----sibling---> DSubTree
                parsed_data_array.append((parent_call_id, edge_type, curr_node_id))

        data_points.append(parsed_data_array)
        done += 1

    except (TooLongLoopingException) as e1:
        ignored_for_loop += 1

    except (TooLongBranchingException) as e2:
        ignored_for_branch += 1

    # print(done)

    # if done % 100000 == 0:
    #     print('Extracted data for {} programs'.format(done), end='\n')
    #     # break

    # randomly shuffle to avoid bias towards initial data points during training
    random.shuffle(data_points)
    raw_data_points = data_points

    config.num_batches = int(len(raw_data_points) / config.batch_size)
    if config.num_batches == 0:
        config.num_batches = 1

    sz = config.num_batches * config.batch_size
    raw_data_points = raw_data_points[:sz]

    # wrangle the evidences and targets into numpy arrays
    nodes = np.zeros((sz, config.decoder.max_ast_depth), dtype=np.int32)
    edges = np.zeros((sz, config.decoder.max_ast_depth), dtype=np.bool)
    targets = np.zeros((sz, config.decoder.max_ast_depth), dtype=np.int32)


    for i, path in enumerate(raw_data_points):
        len_path = min(len(path), config.decoder.max_ast_depth)
        mod_path = path[:len_path]

        nodes[i, :len_path] = [p[0] for p in mod_path]
        edges[i, :len_path] = [p[1] for p in mod_path]
        targets[i, :len_path] = [p[2] for p in mod_path]

    return np.array(nodes), np.array(edges), np.array(targets)


class decoderDict():

    def __init__(self, infer, pre_loaded_vocab=None):
        self.infer = infer
        if not infer:
            self.call_dict = dict()
            self.call_dict['STOP'] = 0
            self.call_count = 1
        else:
            self.call_dict = pre_loaded_vocab.vocab
            self.call_count = pre_loaded_vocab.vocab_size

    def get_or_add_node_val_from_callMap(self, nodeVal):
        if self.infer and (nodeVal not in self.call_dict):
            return None
        elif self.infer or (nodeVal in self.call_dict):
            return self.call_dict[nodeVal]
        else:
            nextOpenPos = self.call_count
            self.call_dict[nodeVal] = nextOpenPos
            self.call_count += 1
            return nextOpenPos

    def get_node_val_from_callMap(self, nodeVal):
        if self.infer and (nodeVal not in self.call_dict):
            return None
        else:
            return self.call_dict[nodeVal]

    def get_call_dict(self):
        return self.call_dict, self.call_count


# Init Tree


class Tree:
    """
    Given the constraints the user specifies, this class will generate a tree that matches them. Since each constraint
    implies that a given piece of evidence must be in or not in the final program, here we focus on generating a tree
    that has all the evidences the user wants and focus on the evidences the user does not want in the MCMC steps.
    """

    def __init__(self, constraints, config):
        self.constraints = constraints
        self.config = config

        # Create a node for each piece of evidence the user wants; each node is linked together by a child edge
        self.candidate = [('DSubTree', CHILD_EDGE)]
        for ev in self.constraints:
            # Skip all evidences that cannot appear in the program
            if not self.constraints[ev]:
                continue

            # Otherwise, create a node and add it to the candidate
            ast = collections.OrderedDict()

            # For now, we are only constraining upon API calls
            ast['node'] = 'DAPICall'
            ast['_call'] = ev

            self.candidate.append((ast, CHILD_EDGE))

    def generate_tree(self):
        """
        Generates a simple AST that fits the user's constraints.
        """


# GRU Tree


class TreeEncoder(object):
    def __init__(self, emb, batch_size, nodes, edges, num_layers, units, depth, output_units):
        cells1 = []
        cells2 = []
        for _ in range(num_layers):
            cells1.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(units))
            cells2.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(units))

        # image of multi-layer RNN:
        # https://hackernoon.com/hn-images/1*6xj691fPWf3S-mWUCbxSJg.jpeg
        self.cell1 = tf.nn.rnn_cell.MultiRNNCell(cells1)
        self.cell2 = tf.nn.rnn_cell.MultiRNNCell(cells2)

        # examples for MultiRNNCell:
        # https://www.programcreek.com/python/example/102269/tensorflow.models.rnn.rnn_cell.MultiRNNCell

        # initial_state has get_shape (batch_size, latent_size), same as psi_mean in the prev code
        curr_state = [tf.truncated_normal([batch_size, units], stddev=0.001)] * num_layers
        curr_out = tf.zeros([batch_size, units])

        # projection matrices for output
        with tf.name_scope("projections"):
            self.projection_w = tf.get_variable('projection_w', [self.cell1.output_size, output_units])
            self.projection_b = tf.get_variable('projection_b', [output_units])

        # get the embedding for each node i
        # embedding_lookup function retrieves rows of the params tensor.
        print('emb: ', emb)
        emb_inp = (tf.nn.embedding_lookup(emb, i) for i in nodes)

        with tf.variable_scope('Tree_network'):

            # the decoder (modified from tensorflow's seq2seq library to fit tree RNNs)
            with tf.variable_scope('rnn'):
                self.state = curr_state
                for i, inp in enumerate(emb_inp):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    with tf.variable_scope('cell1'):  # handles CHILD_EDGE
                        output1, state1 = self.cell1(inp, self.state)
                    with tf.variable_scope('cell2'):  # handles SIBLING EDGE
                        output2, state2 = self.cell2(inp, self.state)

                    # returns either output1 from child_edge or output2 from sibling edge based on edges
                    output = tf.where(edges[i], output1, output2)
                    # if output != 0 then update curr_out, else don't
                    curr_out = tf.where(tf.not_equal(inp, 0), output, curr_out)

                    # update state for each cell in each layer based on which edge we're using
                    self.state = [tf.where(edges[i], state1[j], state2[j]) for j in range(num_layers)]
                    curr_state = [tf.where(tf.not_equal(inp, 0), self.state[j], curr_state[j])
                                  for j in range(num_layers)]

        with tf.name_scope("Output"):
            self.last_output = tf.nn.xw_plus_b(curr_out, self.projection_w, self.projection_b)


# Architecture


# sketch to latent space (normal encoder is evidences to latent space)
class BayesianReverseEncoder(object):
    def __init__(self, config, emb, nodes, edges, infer=False):
        if infer:
            config.reverse_encoder.max_ast_depth = 1

        # self.inputs = [ev.placeholder(config) for ev in config.evidence]
        # print('inputs: ', self.inputs)

        nodes = [nodes[config.reverse_encoder.max_ast_depth - 1 - i] for i in
                 range(config.reverse_encoder.max_ast_depth)]
        edges = [edges[config.reverse_encoder.max_ast_depth - 1 - i] for i in
                 range(config.reverse_encoder.max_ast_depth)]

        # Two halves: one set of NN for calculating the covariance and the other set for calculating the mean
        with tf.variable_scope("Covariance"):
            with tf.variable_scope("APITree"):
                API_Cov_Tree = TreeEncoder(emb, config.batch_size, nodes, edges, config.reverse_encoder.num_layers,
                                           config.reverse_encoder.units, config.reverse_encoder.max_ast_depth,
                                           config.latent_size)
                Tree_Cov = API_Cov_Tree.last_output
                print("Tree Encoder Covariance:", Tree_Cov)

        with tf.variable_scope("Mean"):
            with tf.variable_scope('APITree'):
                API_Mean_Tree = TreeEncoder(emb, config.batch_size, nodes, edges, config.reverse_encoder.num_layers,
                                            config.reverse_encoder.units, config.reverse_encoder.max_ast_depth,
                                            config.latent_size)
                Tree_mean = API_Mean_Tree.last_output
                print("Tree Encoder Mean:", Tree_mean)

            sigmas = Tree_Cov

            # dimension is  3*batch * 1
            finalSigma = tf.layers.dense(tf.reshape(sigmas, [config.batch_size, -1]), config.latent_size,
                                         activation=tf.nn.tanh)
            print('finalsigma 1:', finalSigma)
            finalSigma = tf.layers.dense(finalSigma, config.latent_size, activation=tf.nn.tanh)
            print('finalsigma 2:', finalSigma)
            finalSigma = tf.layers.dense(finalSigma, 1)
            print('finalsigma 3:', finalSigma)

            d = tf.tile(tf.square(finalSigma), [1, config.latent_size])
            d = .00000001 + d
            # denom = d # tf.tile(tf.reshape(d, [-1, 1]), [1, config.latent_size])
            # I = tf.ones([config.batch_size, config.latent_size], dtype=tf.float32)
            self.psi_covariance = d  # I / denom
            print('PSI Covariance: ', d)

            encodings = Tree_mean

            finalMean = tf.layers.dense(tf.reshape(encodings, [config.batch_size, -1]), config.latent_size,
                                        activation=tf.nn.tanh)
            print('finalmean 1:', finalMean)
            finalMean = tf.layers.dense(finalMean, config.latent_size, activation=tf.nn.tanh)
            print('finalmean 2:', finalMean)
            finalMean = tf.layers.dense(finalMean, config.latent_size)
            print('finalmean 3:', finalMean)
            # 4. compute the mean of non-zero encodings
            self.psi_mean = finalMean
            print('PSI Mean:', finalMean)


class BayesianDecoder(object):
    def __init__(self, config, emb, initial_state, nodes, edges):

        cells1, cells2 = [], []
        for _ in range(config.decoder.num_layers):
            cells1.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.decoder.units))
            cells2.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.decoder.units))

        self.cell1 = tf.nn.rnn_cell.MultiRNNCell(cells1)
        self.cell2 = tf.nn.rnn_cell.MultiRNNCell(cells2)

        # placeholders
        self.initial_state = [initial_state] * config.decoder.num_layers
        self.nodes = [nodes[i] for i in range(config.decoder.max_ast_depth)]
        self.edges = [edges[i] for i in range(config.decoder.max_ast_depth)]

        # projection matrices for output
        with tf.variable_scope("projections"):
            self.projection_w = tf.get_variable('projection_w', [self.cell1.output_size,
                                                                 config.decoder.vocab_size])
            self.projection_b = tf.get_variable('projection_b', [config.decoder.vocab_size])
            # tf.summary.histogram("projection_w", self.projection_w)
            # tf.summary.histogram("projection_b", self.projection_b)

        # setup embedding
        emb_inp = (tf.nn.embedding_lookup(emb, i) for i in self.nodes)

        with tf.variable_scope('decoder_network'):
            # the decoder (modified from tensorflow's seq2seq library to fit tree RNNs)
            with tf.variable_scope('rnn'):
                self.state = self.initial_state
                self.outputs = []
                # self.states = []
                for i, inp in enumerate(emb_inp):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    with tf.variable_scope('cell1'):  # handles CHILD_EDGE
                        output1, state1 = self.cell1(inp, self.state)
                    with tf.variable_scope('cell2'):  # handles SIBLING_EDGE
                        output2, state2 = self.cell2(inp, self.state)

                    output = tf.where(self.edges[i], output1, output2)
                    self.state = [tf.where(self.edges[i], state1[j], state2[j])
                                  for j in range(config.decoder.num_layers)]
                    self.outputs.append(output)


# Model


class Model:
    def __init__(self, config, iterator, infer=False):
        assert config.model == 'lle', 'Trying to load different model implementation: ' + config.model
        self.config = config

        newBatch = iterator.get_next()
        nodes, edges, targets = newBatch[:3]

        nodes = tf.transpose(nodes)
        edges = tf.transpose(edges)

        self.nodes = nodes
        self.edges = edges

        with tf.variable_scope("Reverse_Encoder", reuse=tf.AUTO_REUSE):
            embAPI = tf.get_variable('embAPI', [config.reverse_encoder.vocab_size, config.reverse_encoder.units])
            # tf.Print(embAPI)
            self.reverse_encoder = BayesianReverseEncoder(config, embAPI, nodes, edges, infer)
            samples_1 = tf.random_normal([config.batch_size, config.latent_size], mean=0., stddev=1., dtype=tf.float32)

            # get a sample from the latent space
            self.psi_reverse_encoder = self.reverse_encoder.psi_mean + tf.sqrt(
                self.reverse_encoder.psi_covariance) * samples_1

        # setup the decoder with psi as the initial state
        with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
            emb = tf.get_variable('emb', [config.decoder.vocab_size, config.decoder.units])
            lift_w = tf.get_variable('lift_w', [config.latent_size, config.decoder.units])
            lift_b = tf.get_variable('lift_b', [config.decoder.units])

            self.initial_state = tf.nn.xw_plus_b(self.psi_reverse_encoder, lift_w, lift_b, name="Initial_State")
            self.decoder = BayesianDecoder(config, emb, self.initial_state, nodes, edges)

        # get the decoder outputs
        with tf.name_scope("Loss"):
            output = tf.reshape(tf.concat(self.decoder.outputs, 1),
                                [-1, self.decoder.cell1.output_size])
            # what logits are: https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow
            logits = tf.matmul(output, self.decoder.projection_w) + self.decoder.projection_b
            self.ln_probs = tf.nn.log_softmax(logits)
            self.idx = tf.multinomial(logits, 1)

            # 1. generation loss: log P(Y | Z)
            cond = tf.not_equal(tf.reduce_sum(self.reverse_encoder.psi_mean, axis=1), 0)
            cond = tf.reshape(tf.tile(tf.expand_dims(cond, axis=1), [1, config.decoder.max_ast_depth]), [-1])
            cond = tf.where(cond, tf.ones(cond.shape), tf.zeros(cond.shape))

            self.gen_loss = seq2seq.sequence_loss([logits], [tf.reshape(targets, [-1])], [cond])

            # 2. latent loss: regularizer that makes our approximate posterior q(z|x) as similar to p(z|x) as possible
            self.KL_loss = 0.5 * tf.reduce_mean(-tf.log(self.reverse_encoder.psi_covariance)
                                                - 1 + self.reverse_encoder.psi_covariance
                                                + tf.square(-self.reverse_encoder.psi_mean), axis=1)

            self.loss = 0.01 * self.KL_loss + self.gen_loss

            # self.allEvSigmas = [ ev.sigma for ev in self.config.evidence ]
            # unused if MultiGPU is being used
            with tf.name_scope("train"):
                train_ops = get_var_list()['all_vars']

        if not infer:
            opt = tf.train.AdamOptimizer(config.learning_rate)
            self.train_op = opt.minimize(self.loss, var_list=train_ops)

            var_params = [np.prod([dim.value for dim in var.get_shape()])
                          for var in tf.trainable_variables()]
            print('Model parameters: {}'.format(np.sum(var_params)))


# Train


def train(model_settings):
    # Open config file
    print(model_settings.config_path)
    with open(model_settings.config_path) as f:
        config = read_config(json.load(f))

    print(config)
    reader = Reader(model_settings, config)

    # Placeholders for tf data

    nodes_placeholder = tf.placeholder(reader.nodes.dtype, reader.nodes.shape)
    edges_placeholder = tf.placeholder(reader.edges.dtype, reader.edges.shape)
    targets_placeholder = tf.placeholder(reader.targets.dtype, reader.targets.shape)
    # reset batches

    feed_dict = {}
    feed_dict.update({nodes_placeholder: reader.nodes})
    feed_dict.update({edges_placeholder: reader.edges})
    feed_dict.update({targets_placeholder: reader.targets})

    dataset = tf.data.Dataset.from_tensor_slices(
        (nodes_placeholder, edges_placeholder, targets_placeholder))
    batched_dataset = dataset.batch(config.batch_size)
    iterator = batched_dataset.make_initializable_iterator()

    model = Model(config, iterator)

    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
        if model_settings.save_outputs:
            os.makedirs(model_settings.output_model_path, exist_ok=True)
            writer = tf.summary.FileWriter(model_settings.output_model_path)
            writer.add_graph(sess.graph)

        tf.global_variables_initializer().run()

        if model_settings.save_outputs:
            tf.train.write_graph(sess.graph_def, model_settings.output_model_path, 'model.pbtxt')
            tf.train.write_graph(sess.graph_def, model_settings.output_model_path, 'model.pb', as_text=False)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

        # # restore model
        # if clargs.continue_from is not None:
        #     bayou_vars = get_var_list()['all_vars']
        #     old_saver = tf.train.Saver(bayou_vars, max_to_keep=None)
        #     ckpt = tf.train.get_checkpoint_state(clargs.continue_from)
        #     old_saver.restore(sess, ckpt.model_checkpoint_path)

        if model_settings.test_mode:
            os.makedirs(model_settings.output_model_path, exist_ok=True)
            analysis_file = open(os.path.join(model_settings.output_model_path, 'analysis.txt'), 'a+')

        for i in range(config.num_epochs):
            sess.run(iterator.initializer, feed_dict=feed_dict)
            avg_loss, avg_gen_loss, avg_KL_loss = 0., 0., 0.

            for b in range(config.num_batches):
                # run the optimizer
                # loss, _, allEvSigmas = sess.run([model.loss, model.train_op, model.allEvSigmas])
                loss, KL_loss, gen_loss, _ = sess.run([model.loss, model.KL_loss, model.gen_loss, model.train_op])
                avg_loss += np.mean(loss)
                avg_gen_loss += np.mean(gen_loss)
                avg_KL_loss += np.mean(KL_loss)

                step = i * config.num_batches + b + 1
                if step % config.print_step == 0:
                    print_step = '{}/{} (epoch {}) loss: {:.3f}, gen_loss: {:.3f}, KL_loss: {:.3f}, \n\t'\
                            .format(step, config.num_epochs * config.num_batches, i + 1, avg_loss / (b + 1),
                                    avg_gen_loss/(b + 1), avg_KL_loss/(b + 1))

                    print(print_step)

                    if model_settings.test_mode:
                        analysis_file.write(print_step)

                if step % 10000 == 0 and model_settings.save_outputs:
                    checkpoint_dir = os.path.join(model_settings.output_model_path, 'model_temp{}.ckpt'.format(step))
                    saver.save(sess, checkpoint_dir)

            if (i + 1) % config.checkpoint_step == 0:
                if model_settings.save_outputs:
                    checkpoint_dir = os.path.join(model_settings.output_model_path, 'model{}.ckpt'.format(i + 1))
                    saver.save(sess, checkpoint_dir)

                checkpoint_message = 'Model checkpointed: {}. Average for epoch , loss: {:.3f}, gen_loss: {:.3f}, ' \
                                     'KL_loss: {:.3f}'\
                    .format(checkpoint_dir, avg_loss / config.num_batches, avg_gen_loss/config.num_batches,
                            avg_KL_loss/config.num_batches)

                print(checkpoint_message)

                if model_settings.test_mode:
                    analysis_file.write(checkpoint_message)

        if model_settings.save_outputs:
            saver.save(sess, os.path.join(model_settings.output_model_path, 'model_final'))


# Infer


MAX_GEN_UNTIL_STOP = 20
MAX_AST_DEPTH = 5


class TooLongPathError(Exception):
    pass


class IncompletePathError(Exception):
    pass


class InvalidSketchError(Exception):
    pass


class Candidate():
    def __init__(self, initial_state):
        self.tree_currNode = Node("DSubTree")
        self.head = self.tree_currNode

        self.last_item = self.tree_currNode.val
        self.last_edge = SIBLING_EDGE
        self.branch_stack = []

        self.length = 1
        self.log_probabilty = -np.inf
        self.state = initial_state

        self.rolling = True


class BayesianPredictor(object):

    def __init__(self, model_settings, config):
        self.sess = tf.InteractiveSession()
        self.model_settings = model_settings
        self.config = config

        reader = Reader(model_settings, config)

        print("nodes in predictor shape:", reader.nodes.shape)

        # Placeholders for tf data
        nodes_placeholder = tf.placeholder(reader.nodes.dtype, [config.batch_size, 1])
        edges_placeholder = tf.placeholder(reader.edges.dtype, [config.batch_size, 1])
        targets_placeholder = tf.placeholder(reader.targets.dtype, [config.batch_size, 1])

        # reset batches
        feed_dict = {}
        feed_dict.update({nodes_placeholder: reader.nodes})
        feed_dict.update({edges_placeholder: reader.edges})
        feed_dict.update({targets_placeholder: reader.targets})

        dataset = tf.data.Dataset.from_tensor_slices((nodes_placeholder, edges_placeholder, targets_placeholder))
        batched_dataset = dataset.batch(config.batch_size)
        # print("batched dataset:")
        # batched_dataset = tf.Print(batched_dataset, [batched_dataset])
        iterator = batched_dataset.make_initializable_iterator()

        # initialize the model
        config.decoder.max_ast_depth = 1
        self.model = Model(config, iterator, infer=True)

        if model_settings.saved_model_path is None:
            raise ValueError('BayesianPredictor: saved_model_path in ModelSettings may not be None to use Predictor.')

        # restore the saved model
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(model_settings.saved_model_path)
        print("BayesianPredictor: Restoring model from", ckpt.model_checkpoint_path)
        saver.restore(self.sess, ckpt.model_checkpoint_path)

        self.top_k_values, self.top_k_indices = tf.nn.top_k(self.model.ln_probs, k=config.batch_size)

        # [encMean, encCovar] = self.sess.run([self.model.reverse_encoder.psi_mean,
        #                                      self.model.reverse_encoder.psi_covariance], feed_dict)
        # print("Reverse Encoder Mean: ", encMean)
        # print("Reverse Encoder Covariance: ", encCovar)

    def get_state(self, input_nodes, input_edges):
        state = self.model.initial_state
        print(type(state))
        nodes = np.zeros([1, self.config.batch_size])
        edges = np.zeros([1, self.config.batch_size])
        for i in range(len(input_nodes)):
            nodes[0][0] = input_nodes[i]
            edges[0][0] = input_edges[i]
            feed = {self.model.nodes: nodes, self.model.edges: edges}
            # print("Decoding latent space vector into a sketch...")
            state = self.sess.run(tf.convert_to_tensor(state), feed)
        print("mean and cov:", self.get_encoder_mean_variance(nodes, edges))
        return state

    def beam_search(self, input_nodes, input_edges, topK):

        self.config.batch_size = topK

        init_state = self.get_state(input_nodes, input_edges)
        # print("Z Space Mean and Variance with input nodes and edges: ", self.get_encoder_mean_variance(input_nodes.T, input_edges.T))

        candies = [Candidate(init_state[0]) for k in range(topK)]
        candies[0].log_probabilty = -0.0

        i = 0
        while (True):
            # states was batch_size * LSTM_Decoder_state_size
            candies = self.get_next_output_with_fan_out(candies)
            # print([candy.head.breadth_first_search() for candy in candies])
            # print([candy.rolling for candy in candies])

            if self.check_for_all_STOP(candies):  # branch_stack and last_item
                break

            i += 1

            if i == MAX_GEN_UNTIL_STOP:
                break

        candies.sort(key=lambda x: x.log_probabilty, reverse=True)

        return candies

    def check_for_all_STOP(self, candies):
        for candy in candies:
            if candy.rolling == True:
                return False

        return True

    def get_next_output_with_fan_out(self, candies):

        topK = len(candies)

        last_item = [[self.config.decoder.vocab[candy.last_item]] for candy in candies]
        last_edge = [[candy.last_edge] for candy in candies]
        states = [candy.state for candy in candies]

        feed = {}
        feed[self.model.nodes.name] = np.array(last_item, dtype=np.int32).T
        feed[self.model.edges.name] = np.array(last_edge, dtype=np.bool).T
        feed[self.model.initial_state.name] = np.array(states)

        [states, beam_ids, beam_ln_probs, top_idx] = self.sess.run(
            [self.model.decoder.state, self.top_k_indices, self.top_k_values, self.model.idx], feed)

        states = states[0]
        next_nodes = [[self.config.decoder.chars[idx] for idx in beam] for beam in beam_ids]

        log_probabilty = np.array([candy.log_probabilty for candy in candies])
        length = np.array([candy.length for candy in candies])

        for i in range(topK):
            if candies[i].rolling == False:
                length[i] = candies[i].length + 1
            else:
                length[i] = candies[i].length

        for i in range(topK):  # denotes the candidate
            for j in range(topK):  # denotes the items
                if candies[i].rolling == False and j > 0:
                    beam_ln_probs[i][j] = -np.inf
                elif candies[i].rolling == False and j == 0:
                    beam_ln_probs[i][j] = 0.0

        new_probs = log_probabilty[:, None] + beam_ln_probs

        len_norm_probs = new_probs  # / np.power(length[:,None], 1.0)

        rows, cols = np.unravel_index(np.argsort(len_norm_probs, axis=None)[::-1], new_probs.shape)
        rows, cols = rows[:topK], cols[:topK]

        # rows mean which of the original candidate was finally selected
        new_candies = []
        for row, col in zip(rows, cols):
            new_candy = deepcopy(candies[row])  # candies[row].copy()
            if new_candy.rolling:
                new_candy.state = states[row]
                new_candy.log_probabilty = new_probs[row][col]
                new_candy.length += 1

                value2add = next_nodes[row][col]
                # print(value2add)

                if new_candy.last_edge == SIBLING_EDGE:
                    new_candy.tree_currNode = new_candy.tree_currNode.addAndProgressSiblingNode(Node(value2add))
                else:
                    new_candy.tree_currNode = new_candy.tree_currNode.addAndProgressChildNode(Node(value2add))

                # before updating the last item lets check for penultimate value
                if new_candy.last_edge == CHILD_EDGE and new_candy.last_item in ['DBranch', 'DExcept', 'DLoop']:
                    new_candy.branch_stack.append(new_candy.tree_currNode)
                    new_candy.last_edge = CHILD_EDGE
                    new_candy.last_item = value2add

                elif value2add in ['DBranch', 'DExcept', 'DLoop']:
                    new_candy.branch_stack.append(new_candy.tree_currNode)
                    new_candy.last_edge = CHILD_EDGE
                    new_candy.last_item = value2add

                elif value2add == 'STOP':
                    if len(new_candy.branch_stack) == 0:
                        new_candy.rolling = False
                    else:
                        new_candy.tree_currNode = new_candy.branch_stack.pop()
                        new_candy.last_item = new_candy.tree_currNode.val
                        new_candy.last_edge = SIBLING_EDGE
                else:
                    new_candy.last_edge = SIBLING_EDGE
                    new_candy.last_item = value2add

            new_candies.append(new_candy)

        return new_candies

    def get_jsons_from_beam_search(self, nodes, edges, topK):

        candidates = self.beam_search(nodes, edges, topK)

        candidates = [candidate for candidate in candidates if candidate.rolling is False]
        # candidates = candidates[0:1]
        # print(candidates[0].head.breadth_first_search())
        candidate_jsons = [self.paths_to_ast(candidate.head) for candidate in candidates]
        return candidate_jsons

    def paths_to_ast(self, head_node):
        """
        Converts a AST
        :param paths: the set of paths
        :return: the AST
        """
        json_nodes = []
        ast = {'node': 'DSubTree', '_nodes': json_nodes}
        self.expand_all_siblings_till_STOP(json_nodes, head_node.sibling)

        return ast

    def expand_all_siblings_till_STOP(self, json_nodes, head_node):
        """
        Updates the given list of AST nodes with those along the path starting from pathidx until STOP is reached.
        If a DBranch, DExcept or DLoop is seen midway when going through the path, recursively updates the respective
        node type.
        :param nodes: the list of AST nodes to update
        :param path: the path
        :param pathidx: index of path at which update should start
        :return: the index at which STOP was encountered if there were no recursive updates, otherwise -1
        """

        while head_node.val != 'STOP':
            node_value = head_node.val
            astnode = {}
            if node_value == 'DBranch':
                astnode['node'] = node_value
                astnode['_cond'] = []
                astnode['_then'] = []
                astnode['_else'] = []
                self.update_DBranch(astnode, head_node.child)
                json_nodes.append(astnode)
            elif node_value == 'DExcept':
                astnode['node'] = node_value
                astnode['_try'] = []
                astnode['_catch'] = []
                self.update_DExcept(astnode, head_node.child)
                json_nodes.append(astnode)
            elif node_value == 'DLoop':
                astnode['node'] = node_value
                astnode['_cond'] = []
                astnode['_body'] = []
                self.update_DLoop(astnode, head_node.child)
                json_nodes.append(astnode)
            else:
                json_nodes.append({'node': 'DAPICall', '_call': node_value})

            head_node = head_node.sibling

        return

    def update_DBranch(self, astnode, loop_node):
        """
        Updates a DBranch AST node with nodes from the path starting at pathidx
        :param astnode: the AST node to update
        :param path: the path
        :param pathidx: index of path at which update should start
        """
        # self.expand_all_siblings_till_STOP(astnode['_cond'], loop_node, pathidx+1)

        astnode['_cond'] = json_nodes = [{'node': 'DAPICall', '_call': loop_node.val}]
        self.expand_all_siblings_till_STOP(astnode['_then'], loop_node.sibling)
        self.expand_all_siblings_till_STOP(astnode['_else'], loop_node.child)
        return

    def update_DExcept(self, astnode, loop_node):
        """
        Updates a DExcept AST node with nodes from the path starting at pathidx
        :param astnode: the AST node to update
        :param path: the path
        :param pathidx: index of path at which update should start
        """
        self.expand_all_siblings_till_STOP(astnode['_try'], loop_node)
        self.expand_all_siblings_till_STOP(astnode['_catch'], loop_node.child)
        return

    def update_DLoop(self, astnode, loop_node):
        """
        Updates a DLoop AST node with nodes from the path starting at pathidx
        :param astnode: the AST node to update
        :param path: the path
        :param pathidx: index of path at which update should start
        """
        self.expand_all_siblings_till_STOP(astnode['_cond'], loop_node)
        self.expand_all_siblings_till_STOP(astnode['_body'], loop_node.child)
        return

    def get_encoder_mean_variance(self, nodes, edges):
        # setup initial states and feed

        feed = {}
        feed[self.model.nodes] = nodes
        feed[self.model.edges] = edges

        [encMean, encCovar] = self.sess.run(
            [self.model.reverse_encoder.psi_mean, self.model.reverse_encoder.psi_covariance], feed)

        return encMean[0], encCovar[0]

    def random_search(self, nodes, edges):

        # got the state, to be used subsequently
        state = self.get_state(nodes, edges)
        print("state: ", state)
        print("state shape: ", state.shape)
        start_node = Node("DSubTree")
        head, final_state = self.consume_siblings_until_STOP(state, start_node)

        return head.sibling

    def get_prediction(self, node, edge, state):
        feed = {}
        feed[self.model.nodes.name] = np.array([[self.config.decoder.vocab[node]]], dtype=np.int32)
        feed[self.model.edges.name] = np.array([[edge]], dtype=np.bool)
        feed[self.model.initial_state.name] = state
        print("model idx shape:", self.model.idx.shape)
        print("self.model.nodes.shape: ", self.model.nodes.shape)
        print("state len: ", len(self.model.decoder.state))
        [state, idx] = self.sess.run([self.model.decoder.state, self.model.idx], feed)
        idx = idx[0][0]
        state = state[0]
        prediction = self.config.decoder.chars[idx]

        return Node(prediction), state

    def consume_siblings_until_STOP(self, state, init_node):
        # all the candidate solutions starting with a DSubTree node
        head = candidate = init_node
        if init_node.val == 'STOP':
            return head

        stack_QUEUE = []

        while True:

            predictionNode, state = self.get_prediction(candidate.val, SIBLING_EDGE, state)
            candidate = candidate.addAndProgressSiblingNode(predictionNode)

            prediction = predictionNode.val
            if prediction == 'DBranch':
                candidate.child, state = self.consume_DBranch(state)
            elif prediction == 'DExcept':
                candidate.child, state = self.consume_DExcept(state)
            elif prediction == 'DLoop':
                candidate.child, state = self.consume_DLoop(state)
            # end of inner while

            elif prediction == 'STOP':
                break

        # END OF WHILE
        return head, state

    def consume_DExcept(self, state):
        catchStartNode, state = self.get_prediction('DExcept', CHILD_EDGE, state)

        tryStartNode, state = self.get_prediction(catchStartNode.val, CHILD_EDGE, state)
        tryBranch, state = self.consume_siblings_until_STOP(state, tryStartNode)

        catchBranch, state = self.consume_siblings_until_STOP(state, catchStartNode)

        catchStartNode.child = tryStartNode

        return tryBranch, state

    def consume_DLoop(self, state):
        loopConditionNode, state = self.get_prediction('DLoop', CHILD_EDGE, state)
        loopStartNode, state = self.get_prediction(loopConditionNode.val, CHILD_EDGE, state)
        loopBranch, state = self.consume_siblings_until_STOP(state, loopStartNode)

        loopConditionNode.sibling = Node('STOP')
        loopConditionNode.child = loopBranch

        return loopConditionNode, state

    def consume_DBranch(self, state):
        ifStatementNode, state = self.get_prediction('DBranch', CHILD_EDGE, state)
        thenBranchStartNode, state = self.get_prediction(ifStatementNode.val, CHILD_EDGE, state)

        thenBranch, state = self.consume_siblings_until_STOP(state, thenBranchStartNode)
        ifElseBranch, state = self.consume_siblings_until_STOP(state, ifStatementNode)

        #
        ifElseBranch.child = thenBranch

        return ifElseBranch, state


# Plot


def get_ast_paths(js, idx=0):
    # print (idx)
    cons_calls = []
    i = idx
    curr_Node = None
    head = None
    while i < len(js):
        if js[i]['node'] == 'DAPICall':
            cons_calls.append((js[i]['_call'], SIBLING_EDGE))
            if curr_Node == None:
                curr_Node = Node(js[i]['_call'])
                head = curr_Node
            else:
                curr_Node.sibling = Node(js[i]['_call'])
                curr_Node = curr_Node.sibling
        else:
            break
        i += 1
    if i == len(js):
        cons_calls.append(('STOP', SIBLING_EDGE))
        if curr_Node == None:
            curr_Node = Node('STOP')
            head = curr_Node
        else:
            curr_Node.sibling = Node('STOP')
            curr_Node = curr_Node.sibling
        return head, [cons_calls]

    node_type = js[i]['node']

    if node_type == 'DBranch':

        if curr_Node == None:
            curr_Node = Node('DBranch')
            head = curr_Node
        else:
            curr_Node.sibling = Node('DBranch')
            curr_Node = curr_Node.sibling

        nodeC, pC = get_ast_paths(js[i]['_cond'])  # will have at most 1 "path"
        assert len(pC) <= 1
        nodeC_last = nodeC.iterateHTillEnd()
        nodeC_last.sibling, p1 = get_ast_paths(js[i]['_then'])
        nodeE, p2 = get_ast_paths(js[i]['_else'])
        curr_Node.child = Node(nodeC.val, child=nodeE, sibling=nodeC.sibling)

        p = [p1[0] + path for path in p2] + p1[1:]
        pv = [cons_calls + [('DBranch', CHILD_EDGE)] + pC[0] + path for path in p]

        nodeS, p = get_ast_paths(js, i + 1)
        ph = [cons_calls + [('DBranch', SIBLING_EDGE)] + path for path in p]
        curr_Node.sibling = nodeS

        return head, ph + pv

    if node_type == 'DExcept':
        if curr_Node == None:
            curr_Node = Node('DExcept')
            head = curr_Node
        else:
            curr_Node.sibling = Node('DExcept')
            curr_Node = curr_Node.sibling

        nodeT, p1 = get_ast_paths(js[i]['_try'])
        nodeC, p2 = get_ast_paths(js[i]['_catch'])
        p = [p1[0] + path for path in p2] + p1[1:]

        curr_Node.child = Node(nodeT.val, child=nodeC, sibling=nodeT.sibling)
        pv = [cons_calls + [('DExcept', CHILD_EDGE)] + path for path in p]

        nodeS, p = get_ast_paths(js, i + 1)
        ph = [cons_calls + [('DExcept', SIBLING_EDGE)] + path for path in p]
        curr_Node.sibling = nodeS
        return head, ph + pv

    if node_type == 'DLoop':
        if curr_Node == None:
            curr_Node = Node('DLoop')
            head = curr_Node
        else:
            curr_Node.sibling = Node('DLoop')
            curr_Node = curr_Node.sibling
        nodeC, pC = get_ast_paths(js[i]['_cond'])  # will have at most 1 "path"
        assert len(pC) <= 1
        nodeC_last = nodeC.iterateHTillEnd()
        nodeC_last.sibling, p = get_ast_paths(js[i]['_body'])

        pv = [cons_calls + [('DLoop', CHILD_EDGE)] + pC[0] + path for path in p]
        nodeS, p = get_ast_paths(js, i + 1)
        ph = [cons_calls + [('DLoop', SIBLING_EDGE)] + path for path in p]

        curr_Node.child = nodeC
        curr_Node.sibling = nodeS

        return head, ph + pv


# ---------------------------------- FIX PLOT CODE -------------------------------------------
def plot(model_settings):
    with open(model_settings.config_path) as f:
        config = read_config(json.load(f))

    config.batch_size = 1

    predictor = BayesianPredictor(model_settings, config)

    # Plot with all Evidences
    with open(model_settings.data_path, 'rb') as f:
        deriveAndScatter(model_settings, config, f, predictor)
    #
    # with open(clargs.input_file[0], 'rb') as f:
    #     useAttributeAndScatter(f, 'b2')


def useAttributeAndScatter(model_settings, f, att, max_nums=10000):
    psis = []
    labels = []
    item_num = 0
    for program in ijson.items(f, 'programs.item'):
        api_call = get_api(get_calls_from_ast(program['ast']['_nodes']))
        if api_call != 'N/A':
            labels.append(api_call)
            if att not in program:
                return
            psis.append(program[att])
            item_num += 1

        if item_num > max_nums:
            break

    psis = np.array(psis)
    name = "RE" if att == "b2" else att
    fitTSEandplot(psis, labels, name)


def deriveAndScatter(model_settings, config, f, predictor, max_nums=10000):
    psis = []
    labels = []
    item_num = 0
    for program in ijson.items(f, 'programs.item'):
        # print("program: ", program)
        shortProgram = {'ast': program['ast']}
        # print("short program:", shortProgram)
        api_call = get_api(get_calls_from_ast(shortProgram['ast']['_nodes']))
        # print("nodes:", nodes)
        # print("edges:", edges)
        if api_call != 'N/A':
            labels.append(api_call)
            nodes, edges, _ = read_program_from_json(config, program)
            psis.append(predictor.get_encoder_mean_variance(nodes, edges)[0])
            item_num += 1

        if item_num > max_nums:
            break

    # print(np.array(psis).shape)
    # print(psis[0])
    fitTSEandplot(model_settings, psis, labels)


def fitTSEandplot(model_settings, psis, labels):
    model = TSNE(n_components=2, init='random', verbose=1, perplexity=100)
    psis_2d = model.fit_transform(psis)
    assert len(psis_2d) == len(labels)
    scatter(model_settings, zip(psis_2d, labels))


def get_api(calls):
    calls = [call.replace('$NOT$', '') for call in calls]
    # print("calls:", calls)
    apis = [[re.findall(r"[\w']+", call)[:3]] for call in calls]
    # print("apis 1:", apis)
    apis = [call for _list in apis for calls in _list for call in calls]
    # print("apis 2:", apis)
    label = "N/A"
    guard = []
    for api in apis:
        if api in ['xml', 'sql', 'crypto', 'awt', 'swing', 'security', 'net', 'math', 'String', 'Arrays', 'ArrayList']:
            label = api
            guard.append(label)

    if len(set(guard)) != 1:
        return 'N/A'
    else:
        return guard[0]


def scatter(model_settings, data):
    dic = {}
    for psi_2d, label in data:
        if label == 'N/A':
            continue
        if label not in dic:
            dic[label] = []
        dic[label].append(psi_2d)

    labels = list(dic.keys())
    labels.sort(key=lambda l: len(dic[l]), reverse=True)
    #     for label in labels[clargs.top:]:
    #         del dic[label]

    labels = dic.keys()
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(dic)))
    plotpoints = []
    for label, color in zip(labels, colors):
        x = list(map(lambda s: s[0], dic[label]))
        y = list(map(lambda s: s[1], dic[label]))
        plotpoints.append(plt.scatter(x, y, color=color))

    if model_settings.save_outputs:
        os.makedirs(model_settings.output_plots_path, exist_ok=True)

        plt.legend(plotpoints, labels, scatterpoints=1, loc='lower left', ncol=3, fontsize=12)
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.savefig(model_settings.output_plots_path + "tSNE" + ".png", bbox_inches='tight')

def get_calls_from_ast(ast):
    calls = []
    _, ast_paths = get_ast_paths(ast)
    for path in ast_paths:
        calls += [call[0] for call in path]
    return calls


def sample_test(model_settings, use_beam_search):

    with open(model_settings.data_path) as file:
        program_dict = json.load(file)

    with open(model_settings.config_path) as f:
        config = read_config(json.load(f))
        nodes, edges = read_input_program(model_settings, config, model_settings.data_path)

    if use_beam_search:
        config.batch_size = 20
    else:
        config.batch_size = 1

    print("Loading Bayou predictor...")
    predictor = BayesianPredictor(model_settings, config)
    print("Bayou predictor loaded!")

    # list of dictionaries (each prog is a dict)
    programs = program_dict['programs']
    print(programs)


    # BEAM SEARCH
    if (use_beam_search):
        #         candies = predictor.beam_search(programs, topK=config.batch_size)
        print(config.batch_size)
        candies = predictor.beam_search(nodes.T, edges.T, topK=config.batch_size)

        best_path = None
        best_prob = 100000000
        for i, candy in enumerate(candies):
            path = candy.head.depth_first_search()
            prob = candy.log_probabilty
            # print(prob)
            if best_prob > prob:
                best_path = path
                best_prob = prob
            #
            # dot = plot_path(model_settings, i, path, prob)
        #     print(path)
        #     # print()
        print("best path: ", best_path)
        print("best prob: ", best_prob)
        jsons = predictor.get_jsons_from_beam_search(nodes.T, edges.T, topK=config.batch_size)

        print(jsons)

        # with open('asts/output_' + 'sample_program' + FILENAME + '.json', 'w') as f:
        #     json.dump({'programs': programs, 'asts': jsons}, fp=f, indent=2)

    else:  # breadth-first search
        path_head = predictor.random_search(nodes.T, edges.T)
        path = path_head.depth_first_search()

        randI = random.randint(0, 1000)
        dot = plot_path(model_settings, randI, path, 1.0)
        print(randI)
        print(path)

    return
