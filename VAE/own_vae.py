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
import bayou_vae_functions as bayou


class VAE:
    def __init__(self, model_settings):
        self.model_settings = model_settings

        # Create config object from config file
        with open(self.model_settings.config_path) as f:
            self.config = bayou.read_config(json.load(f))

        # Read in nodes and edges from input data
        reader = bayou.Reader(model_settings, self.config)
        self.nodes = reader.nodes
        self.edges = reader.edges

    # def train(self):
        # Create placeholders for tf data
        nodes_placeholder = tf.placeholder(self.nodes.dtype, self.nodes.shape)
        edges_placeholder = tf.placeholder(self.edges.dtype, self.edges.shape)

        # Initialize feed dictionary
        feed_dict = {nodes_placeholder: self.nodes, edges_placeholder: self.edges}

        # Batch input data
        dataset = tf.data.Dataset.from_tensor_slices(
            (nodes_placeholder, edges_placeholder))
        batched_dataset = dataset.batch(self.config.batch_size, drop_remainder=True)
        batch_iterator = batched_dataset.make_initializable_iterator()

    #     # Get VAE model
    #     model = self.get_model(batch_iterator)
    #
    # def get_model(self, batch_iterator):

        # Get nodes and edges in batch
        new_batch = batch_iterator.get_next()
        batch_nodes, batch_edges = new_batch[:2]
        nodes = tf.transpose(batch_nodes)
        edges = tf.transpose(batch_edges)

        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
            encoder_embedding = tf.get_variable('emb')









