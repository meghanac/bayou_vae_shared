from __future__ import print_function

import collections

import numpy as np
import torch

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
from sklearn.manifold import TSNE

import dataset
from dataset import ModelSettings, Node, Dataset, CHILD_EDGE, SIBLING_EDGE
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
from tensorboardX import SummaryWriter

'''

Steps:
1. Load Dataset
2. Make Dataset Iterable
3. Create Model Class
4. Instantiate Model Class
5. Instantiate Loss Class
6. Instantiate Optimizer Class
7. Train Model

'''

train_model_settings = ModelSettings(dataset.SMALL_SPLIT_TRAINING_DATASET_PATH,
                                     dataset.SMALL_TRAINING_DATASET_CONFIG_PATH,
                                     True,
                                     test_mode=False)
validation_model_settings = ModelSettings(dataset.SMALL_VALIDATION_DATASET_PATH,
                                          dataset.SMALL_TRAINING_DATASET_CONFIG_PATH,
                                          False,
                                          test_mode=False)
# train_model_settings = ModelSettings(dataset.TINY_SPLIT_TRAINING_DATASET_PATH,
#                                      dataset.TINY_TRAINING_DATASET_CONFIG_PATH,
#                                      True,
#                                      test_mode=False)
# validation_model_settings = ModelSettings(dataset.TINY_VALIDATION_DATASET_PATH,
#                                           dataset.TINY_TRAINING_DATASET_CONFIG_PATH,
#                                           False,
#                                           test_mode=False)



# train_model_settings = ModelSettings(dataset.MED_TRAINING_DATASET_PATH, dataset.MED_TRAINING_DATASET_CONFIG_PATH,
#                                      False,
#                                      test_mode=False)
# config, train_nodes, train_edges, train_targets = dataset.extract_data_create_datasets(train_model_settings)

# # TODO: change this once testing functions work
# program_number = 1
# sample_program = 'sample_program' + str(program_number) + '.json'
# data_path = os.path.join(dataset.TEST_DATA_PATH, sample_program)
#
# print(data_path)

# TODO: uncomment when you actually implement this
# test_vae(data_path, config_path, saved_model_path, save_outputs, test_mode)

# TODO: add saved model path
# test_model_settings = ModelSettings(
#     "/Users/meghanachilukuri/Documents/GitHub/Jermaine-Research/Research/Code/Model_Iterations/11:20/work/save/config.json",
#     '/Users/meghanachilukuri/Documents/GitHub/Jermaine-Research/Research/Code/Model_Iterations/11:20/work/save/config.json', False,
#     test_mode=True)

# _, test_nodes, test_edges, test_targets = dataset.extract_data_create_datasets(test_model_settings)

training_ast_dataset = dataset.ASTDataset(train_model_settings)
validation_ast_dataset = dataset.ASTDataset(validation_model_settings)

'''
================================================ 

STEP 2 : MAKE DATASET ITERABLE

================================================
'''

# train_batch_size = config.batch_size
# num_epochs = config.num_epochs
#
# train_nodes_loader = torch.utils.data.DataLoader(dataset=train_nodes,
#                                                  batch_size=train_batch_size,
#                                                  shuffle=False)
# train_edges_loader = torch.utils.data.DataLoader(dataset=train_edges,
#                                                  batch_size=train_batch_size,
#                                                  shuffle=False)
# train_targets_loader = torch.utils.data.DataLoader(dataset=train_targets,
#                                                    batch_size=train_batch_size,
#                                                    shuffle=False)

# test_batch_size = 1
# test_nodes_loader = torch.utils.data.DataLoader(dataset=test_nodes,
#                                                 batch_size=test_batch_size,
#                                                 shuffle=False)
# test_edges_loader = torch.utils.data.DataLoader(dataset=test_edges,
#                                                 batch_size=test_batch_size,
#                                                 shuffle=False)
# test_targets_loader = torch.utils.data.DataLoader(dataset=test_targets,
#                                                   batch_size=test_batch_size,
#                                                   shuffle=False)

train_batch_size = training_ast_dataset.get_config().batch_size

'''
================================================ 

STEP 3 : CREATE MODEL CLASS

================================================
'''


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


class TreeRNN(nn.Module):
    """
    TreeRNN based on Top-down Tree LSTM Networks (Zhang et. al.)
    Updates either Child RNN or Sibling RNN based on production path.
    Is used for both encoder and decoder.
    """

    def __init__(self, config, batch_size, input_dim, hidden_units, num_layers, emb):
        super(TreeRNN, self).__init__()

        self.config = config

        self.batch_size = batch_size

        # Hidden units dimensions
        self.hidden_units = hidden_units

        # Number of hidden layers
        self.num_layers = num_layers

        # GRU to handle child edges
        self.child_rnn = nn.GRU(input_dim, hidden_units, num_layers, batch_first=True)

        # GRU to handle sibling edges
        self.sibling_rnn = nn.GRU(input_dim, hidden_units, num_layers, batch_first=True)

        # Initialize state attributes
        self.initial_state = torch.zeros([num_layers, batch_size, self.hidden_units])
        self.state = torch.zeros([num_layers, batch_size, self.hidden_units])

        # Embedding
        self.emb = emb

        # Outputs
        self.outputs = []

    def forward(self, nodes, edges, initial_state):
        # Create generator for node embeddings
        emb_inp = (self.emb(i.to(dtype=torch.long)) for i in nodes)  # size = [batch size, max ast depth, num_units]

        # Initialize states and outputs
        self.initial_state = initial_state  # TODO: figure out if we need to reset state before every sequence or not (not in Bayou)
        self.state = initial_state
        self.outputs = []
        curr_out = torch.zeros([self.batch_size, self.hidden_units])
        curr_state = self.state

        # Feed in each program (embedded nodes) in the batch
        for i, inp in enumerate(emb_inp):
            # Resize input. Input needs to be of size (seq_len, batch, input_dim) which should be
            # (max_ast_depth, batch_size, embedding_size) and embedding_size == num_units
            inp = inp.view(self.config.decoder.max_ast_depth, 1, self.config.decoder.units)

            # Feed input into Child and Sibling GRUs
            output_c, state_c = self.child_rnn(inp, self.state)
            output_s, state_s = self.sibling_rnn(inp, self.state)

            # Select outputs based on whether there's a child edge (True) or sibling edge (False)
            output = torch.where(edges[i], output_c.T, output_s.T)
            output = output.T
            # Update outputs
            self.outputs.append(output)
            curr_out = torch.where(torch.ne(inp, 0), output, curr_out)

            # Update states
            # Note: To convert a list of tensors to a tensor, use torch.stack
            #       (https://discuss.pytorch.org/t/how-to-turn-a-list-of-tensor-to-tensor/8868/5)
            self.state = torch.stack(
                [torch.where(edges[i].view(self.config.decoder.max_ast_depth, 1), state_c[j], state_s[j]) for j in
                 range(self.num_layers)])
            curr_state = torch.stack(
                [torch.where(torch.ne(inp, 0), self.state[j], curr_state[j]) for j in range(self.num_layers)])

        return torch.stack(self.outputs), self.state


class VRNN(nn.Module):
    """
    Variational Autoencoder
    """

    def __init__(self, model_settings, config):
        super(VRNN, self).__init__()

        self.config = config

        self.max_ast_length = config.decoder.max_ast_depth
        self.latent_size = config.latent_size
        self.num_layers = config.decoder.num_layers
        self.hidden_size = config.decoder.units

        # Set up embedding for nodes
        vocab_size = config.decoder.vocab_size
        embedding_size = config.decoder.units
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        batch_size = config.batch_size

        # states need to be of size (num_layers * num_directions, batch, hidden_size)
        self.encoder = TreeRNN(config, batch_size, embedding_size, self.hidden_size, self.num_layers, self.embedding)
        self.decoder = TreeRNN(config, batch_size, embedding_size, self.hidden_size, self.num_layers, self.embedding)

        self.hidden2mean = nn.Linear(self.hidden_size * self.num_layers, self.latent_size)
        self.hidden2logv = nn.Linear(self.hidden_size * self.num_layers, self.latent_size)
        self.latent2hidden = nn.Linear(self.latent_size, self.hidden_size * self.num_layers)
        self.outputs2vocab = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, nodes, edges):
        # ENCODER

        # Initialize Encoder state to normal distribution
        # TODO: see if we can sample from truncated normal instead
        enc_initial_state = torch.empty([self.num_layers, self.config.decoder.max_ast_depth, self.hidden_size]).normal_(
            mean=0, std=0.001)

        _, hidden = self.encoder(nodes, edges, enc_initial_state)

        if self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(self.config.batch_size, self.hidden_size * self.num_layers)
        else:
            # removes all the dimensions that are 1
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([self.config.decoder.max_ast_depth, self.latent_size]))
        z = z * std + mean

        # DECODER

        # Get initial hidden state for decoder
        hidden = self.latent2hidden(z)
        if self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, self.config.batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # TODO: test to see if we want to implement dropout
        # # decoder input
        # if self.word_dropout_rate > 0:
        #     # randomly replace decoder input with <unk>
        #     prob = torch.rand(input_sequence.size())
        #     if torch.cuda.is_available():
        #         prob = prob.cuda()
        #     prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
        #     decoder_input_sequence = input_sequence.clone()
        #     decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
        #     input_embedding = self.embedding(decoder_input_sequence)
        # input_embedding = self.embedding_dropout(input_embedding)
        # packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        # TODO: make sure this all works when batch size = 1
        outputs, _ = self.decoder(nodes, edges, hidden)
        outputs = outputs.squeeze()

        # project outputs to vocab
        b, s, _ = outputs.size()
        logp = nn.functional.log_softmax(self.outputs2vocab(outputs.view(-1, outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp, mean, logv, z

    def inference(self, n=4, z=None):
        """

        :param n: batch size
        :param z: Variable of size (batch_size, latent_size) that represents latent space Z
        :return: samples, z
        """

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.num_layers, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
        sequence_running = torch.arange(0, batch_size,
                                        out=self.tensor()).long()  # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size,
                                    out=self.tensor()).long()  # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).long()

        t = 0
        while (t < self.max_sequence_length and len(running_seqs) > 0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update global running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:, t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
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
    def check_for_all_STOP(self, candies):
        for candy in candies:
            if candy.rolling == True:
                return False

        return True

'''
================================================ 

STEP 4-7 : TRAIN THE MODEL

================================================
'''
# TODO: fix this
def idx2word(idx, i2w):

    sent_str = [str()]*len(idx)

    for i, sent in enumerate(idx):

        for word_id in sent:

            sent_str[i] += i2w[str(word_id)] + " "

        sent_str[i] = sent_str[i].strip()

    return sent_str


def train(model_settings, config, training_dataset, validation_dataset=None):
    """

    :param model_settings:
    :param config:
    :param training_dataset:
    :param validation_dataset:
    :return:
    """
    # Initialize model
    model = VRNN(model_settings, config)

    # Initialize datasets
    splits = ['train', 'valid']
    dataset = collections.OrderedDict()
    dataset['train'] = training_dataset
    dataset['valid'] = validation_dataset

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    # Save logging information if required
    if model_settings.save_outputs and not model_settings.test_mode:
        os.makedirs(model_settings.output_model_path, exist_ok=True)
        writer = SummaryWriter(model_settings.output_model_path)
        writer.add_text("model", str(model))
        writer.add_text("model settings:", str(model_settings))

        analysis_file = open(os.path.join(model_settings.output_model_path, 'analysis.txt'), 'a+')
        analysis_file.write("model: " +  str(model) + "\n")
        analysis_file.write("model settings: " + str(model_settings) + "\n")

    # TODO: re-implement
    # def kl_anneal_function(anneal_function, step, k, x0):
    #     if anneal_function == 'logistic':
    #         return float(1 / (1 + np.exp(-k * (step - x0))))
    #     elif anneal_function == 'linear':
    #         return min(1, step / x0)

    def kl_anneal_function(step, x0):
        """
        Linear anneal function for KL Divergence
        Based on findings presented in paper "Generating Sentences from a Continuous Space" by Bowman et al 2015
        https://arxiv.org/abs/1511.06349

        :param (int) step: step in training
        :param (int) x0: step on which KL weight = 1
        :return: KL weight
        """
        return min(1, step / x0)

    # Initialize loss function
    NLL = torch.nn.NLLLoss() # TODO: see if we want to do NLLLoss(size_average=False)

    def loss_fn(logp, target, mean, logv, step, x0):
        """
        Calculates generation loss, KL divergence and KL weight for a given step

        :param logp: # TODO: fill out
        :param target: target
        :param mean: mean of Z
        :param logv: # TODO: fill out
        :param step: step in training
        :param x0: step on which KL weight is equal to 1
        :return: generation loss (NLL_loss), KL divergence (KL_loss), KL weight
        """

        target = target.to(dtype=torch.long)

        # flatten
        target = target.view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(step, x0)

        return NLL_loss, KL_loss, KL_weight

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Initialize variables
    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    train_losses = []
    valid_losses = []

    for epoch in range(config.num_epochs):
        # Initialize running loss variables calculated in epoch for plotting purposes
        train_running_loss = 0.0
        valid_running_loss = 0.0

        for split in splits:
            # Load the data
            data_loader = torch.utils.data.DataLoader(dataset=dataset[split],
                                                      batch_size=config.batch_size, shuffle=(split == 'train'),
                                                      drop_last=True)

            tracker = collections.defaultdict(tensor)

            # Train or Evaluate the model
            if split == 'train':
                model.train()
            else:
                model.eval()

            for iteration, batch in enumerate(data_loader):

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                logp, mean, logv, z = model(batch['node'], batch['edge'])

                # loss calculation
                NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'], mean, logv, step, 2500)

                loss = (NLL_loss + KL_weight * KL_loss) / config.batch_size

                # Update running losses
                if split == 'train':
                    train_running_loss += loss.item() * batch['node'].size(0)
                else:
                    valid_running_loss += loss.item() * batch['node'].size(0)

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                # Book keeping
                # Note: need to unsqueeze loss to turn a 0-dim tensor to a 1-dim tensor
                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.unsqueeze(0)))

                if model_settings.save_outputs and not model_settings.test_mode:
                    writer.add_scalar("%s/ELBO" % split.upper(), loss.item(), epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/NLL Loss" % split.upper(), NLL_loss.item() / config.batch_size,
                                      epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Loss" % split.upper(), KL_loss.item() / config.batch_size,
                                      epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Weight" % split.upper(), KL_weight, epoch * len(data_loader) + iteration)

                if iteration % config.print_step == 0 or iteration + 1 == len(data_loader):
                    print_message = '{}, Batch {:04d}/{}, Loss {:9.4f}, NLL-Loss {:9.4f}, KL-Loss {:9.4f}, KL-Weight {:6.3f}'.format(
                              split, iteration, len(data_loader) - 1, loss.item(), NLL_loss.item() / config.batch_size,
                              KL_loss.item() / config.batch_size, KL_weight)
                    print(print_message)
                    if model_settings.save_outputs and not model_settings.test_mode:
                        analysis_file.write(print_message + "\n")

                if split == 'valid':
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    # tracker['target_sents'] += idx2word(batch['target'].data, i2w=dataset['train'].get_i2w())
                    tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

            epoch_print_message = '{}, Epoch {:02d}/{}, Mean ELBO {:9.4f}'.format(split, epoch, config.num_epochs, torch.mean(tracker['ELBO']))
            print(epoch_print_message)

            if model_settings.save_outputs and not model_settings.test_mode:
                analysis_file.write(epoch_print_message + "\n")

            if split == 'train':
                epoch_loss = train_running_loss / len(dataset['train'])
                train_losses.append(epoch_loss)
            else:
                epoch_loss = valid_running_loss / len(dataset['valid'])
                valid_losses.append(epoch_loss)

            if model_settings.save_outputs and not model_settings.test_mode:
                writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                dump = {'target_sents': tracker['target_sents'], 'z': tracker['z'].tolist()}
                os.makedirs(model_settings.output_data_path, exist_ok=True)
                with open(model_settings.output_data_path + model_settings.time + '_valid_E%i.json' % epoch, 'w') as dump_file:
                    json.dump(dump, dump_file)

            # save checkpoint
            if split == 'train' and model_settings.save_outputs and not model_settings.test_mode:
                f = model_settings.output_model_path + "saved_model"
                torch.save(model.state_dict(), f)
                print("Model saved at %s" % f)

    print("Finished training model")

    if model_settings.save_outputs and not model_settings.test_mode:
        os.makedirs(model_settings.output_plots_path, exist_ok=True)
        print("epochs: ", np.array(range(config.num_epochs)))
        print("training losses: ", np.array(train_losses))
        plt.plot(np.array(range(config.num_epochs)), np.array(train_losses), label='Training Loss')
        plt.plot(np.array(range(config.num_epochs)), np.array(valid_losses), label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(model_settings.output_plots_path + "train_val_loss.png", bbox_inches='tight')


train(train_model_settings, training_ast_dataset.get_config(), training_ast_dataset,
      validation_dataset=validation_ast_dataset)

