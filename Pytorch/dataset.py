import argparse
import json
import os
import os.path
import pickle
import random
from collections import defaultdict

import numpy as np

import ijson
import pandas as pd
import torch
import torch.utils.data as data
from torchvision.datasets.folder import default_loader

from datetime import datetime

CONFIG_GENERAL = ['model', 'latent_size', 'batch_size', 'num_epochs',
                  'learning_rate', 'print_step', 'checkpoint_step']
CONFIG_ENCODER = ['name', 'units', 'num_layers', 'tile', 'max_depth', 'max_nums', 'ev_drop_prob', 'ev_call_drop_prob']
CONFIG_DECODER = ['units', 'num_layers', 'max_ast_depth']
CONFIG_REVERSE_ENCODER = ['units', 'num_layers', 'max_ast_depth']
CONFIG_INFER = ['vocab', 'vocab_size']

PYTHON_RECURSION_LIMIT = 10000

TINY_TRAINING_DATASET_PATH = '../VAE/data/data.json'
TINY_TRAINING_DATASET_CONFIG_PATH = '../VAE/config/tiny_config.json'
LARGE_TRAINING_DATASET_PATH = '../VAE/data/DATA-extracted-for-CACM-train.json'
LARGE_TRAINING_DATASET_CONFIG_PATH = '../VAE/config/large_dataset_config.json'
TEST_DATA_PATH = '../VAE/data/test_data/'
MED_TRAINING_DATASET_PATH = "../VAE/data/training_data-500000.json"
MED_TRAINING_DATASET_CONFIG_PATH = '../VAE/config/med_dataset_config.json'
SMALL_TRAINING_DATASET_PATH = "../VAE/data/training_data-100k.json"
SMALL_TRAINING_DATASET_CONFIG_PATH = '../VAE/config/small_dataset_config.json'
SMALL_SPLIT_TRAINING_DATASET_PATH = "../VAE/data/small_training-0.8_100k.json"
SMALL_VALIDATION_DATASET_PATH = "../VAE/data/small_validation-0.2_100k.json"
TINY_SPLIT_TRAINING_DATASET_PATH = "../VAE/data/tiny_training-0.8_1400.json"
TINY_VALIDATION_DATASET_PATH = "../VAE/data/tiny_validation-0.2_1400.json"



class ModelSettings:
    """
    Object that is passed through functions with information like paths of input data and config and on parts of model
    should be saved and where.
    """

    def __init__(self, data_path, config_path, save_outputs, saved_model_path=None, test_mode=False, validation_data_path=None, test_data_path=None):
        self.data_path = data_path
        self.config_path = config_path

        now = datetime.now()
        datetime_of_run = now.strftime("%m-%d-%y_%H:%M:%S")
        self.time = datetime_of_run

        self.save_outputs = save_outputs
        if save_outputs:
            self.output_ast_path = 'output/' + datetime_of_run + '/asts/'
            self.output_data_path = 'output/' + datetime_of_run + '/data/'
            self.output_model_path = 'output/' + datetime_of_run + '/models/'
            self.output_plots_path = 'output/' + datetime_of_run + '/plots/'
            self.output_config_path = 'output/' + datetime_of_run + '/configs/'

        if saved_model_path is not None:
            self.saved_model_path = saved_model_path

        self.test_mode = test_mode

        if validation_data_path is not None:
            self.validation_data_path = validation_data_path

        if test_data_path is not None:
            self.test_data_path = test_data_path


def read_config(js):
    '''
    Converts json into config object
    :param js: json
    :return: config
    '''
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


CHILD_EDGE = True
SIBLING_EDGE = False

MAX_LOOP_NUM = 3
MAX_BRANCHING_NUM = 3


class Candidate:
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


class TooLongLoopingException(Exception):
    pass


class TooLongBranchingException(Exception):
    pass


class Node:
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


class ASTDataset(data.Dataset):
    """
    Custom Dataset class for PyTorch VRNN
    """

    def __init__(self, model_settings, validation=False, test=False):
        self.model_settings = model_settings
        with open(model_settings.config_path) as f:
            config = read_config(json.load(f))
            self.config = config

        random.seed(12)

        data_points = []
        done, ignored_for_branch, ignored_for_loop = 0, 0, 0
        infer = False
        self.decoder_api_dict = decoderDict(infer, self.config.decoder)

        assert(not (validation and test))

        if validation:
            assert(model_settings.validation_data_path is not None)
            f = open(model_settings.validation_data_path, 'rb')
        elif test:
            assert(model_settings.test_data_path is not None)
            f = open(model_settings.test_data_path, 'rb')
        else:
            f = open(model_settings.data_path, 'rb')

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
        if validation:
            dataset_type = "validation"
        elif test:
            dataset_type = "testing"
        else:
            dataset_type = "training"
        print('{:8d} programs/asts in {} data'.format(done, dataset_type))
        print('{:8d} programs/asts missed in {} data for loop'.format(ignored_for_loop, dataset_type))
        print('{:8d} programs/asts missed in {} data for branch'.format(ignored_for_branch, dataset_type))

        # randomly shuffle to avoid bias towards initial data points during training
        random.shuffle(data_points)

        raw_data_points = data_points

        sz = len(raw_data_points)

        # wrangle the evidences and targets into numpy arrays
        self.nodes = np.zeros((sz, config.decoder.max_ast_depth), dtype=np.int32)
        self.edges = np.zeros((sz, config.decoder.max_ast_depth), dtype=np.bool)
        self.targets = np.zeros((sz, config.decoder.max_ast_depth), dtype=np.int32)

        data = defaultdict(dict)

        for i, path in enumerate(raw_data_points):
            len_path = min(len(path), config.decoder.max_ast_depth)
            mod_path = path[:len_path]

            # print(mod_path)
            id = len(data)
            self.nodes[i, :len_path] = [p[0] for p in mod_path]
            self.edges[i, :len_path] = [p[1] for p in mod_path]
            self.targets[i, :len_path] = [p[2] for p in mod_path]

        # if model_settings.save_outputs and not model_settings.test_mode:
        #     os.makedirs(model_settings.output_data_path, exist_ok=True)
        #     with open(model_settings.output_data_path + 'nodes_edges_targets.txt', 'wb') as f:
        #         pickle.dump([self.nodes, self.edges, self.targets], f)
        #
        #     jsconfig = dump_config(config)
        #     with open((model_settings.output_data_path + 'config.json'), 'w') as f:
        #         json.dump(jsconfig, fp=f, indent=2)

        if model_settings.save_outputs and not model_settings.test_mode:
            os.makedirs(model_settings.output_model_path, exist_ok=True)
            with open((model_settings.output_model_path + 'analysis.txt'), 'w') as f:
                f.write("data path: " + model_settings.data_path + "\n")
                f.write("config path: " + model_settings.config_path + "\n")
                f.write("latent size: " + str(config.latent_size) + "\n")
                f.write("batch size: " + str(config.batch_size) + "\n")
                f.write("num epochs: " + str(config.num_epochs) + "\n")
                f.write("learning rate: " + str(config.learning_rate) + "\n")
                f.write("print step: " + str(config.print_step) + "\n")
                f.write("checkpoint step: " + str(config.checkpoint_step) + "\n")
                f.write("encoder/decoder hidden units: " + str(config.decoder.units) + "\n")
                f.write("encoder/decoder num layers: " + str(config.decoder.num_layers) + "\n")
                f.write("max AST depth: " + str(config.decoder.max_ast_depth) + "\n")
                f.write("vocab size: " + str(config.decoder.vocab_size) + "\n")

    def __getitem__(self, index):
        return {'node': self.nodes[index], 'edge': self.edges[index], 'target': self.targets[index]}

    def __len__(self):
        return len(self.nodes)

    def get_config(self):
        return self.config

