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
from dataset import ModelSettings
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils

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

'''
================================================

STEP 1 : LOAD DATASET 

================================================
'''

train_model_settings = ModelSettings(dataset.SMALL_TRAINING_DATASET_PATH, dataset.SMALL_TRAINING_DATASET_CONFIG_PATH,
                                     False,
                                     test_mode=False)
# train_model_settings = ModelSettings(dataset.MED_TRAINING_DATASET_PATH, dataset.MED_TRAINING_DATASET_CONFIG_PATH,
#                                      False,
#                                      test_mode=False)
config, train_nodes, train_edges, train_targets = dataset.extract_data_create_datasets(train_model_settings)

# TODO: change this once testing functions work
program_number = 1
sample_program = 'sample_program' + str(program_number) + '.json'
data_path = os.path.join(dataset.TEST_DATA_PATH, sample_program)

print(data_path)

# TODO: uncomment when you actually implement this
# test_vae(data_path, config_path, saved_model_path, save_outputs, test_mode)

# TODO: add saved model path
# test_model_settings = ModelSettings(
#     "/Users/meghanachilukuri/Documents/GitHub/Jermaine-Research/Research/Code/Model_Iterations/11:20/work/save/config.json",
#     '/Users/meghanachilukuri/Documents/GitHub/Jermaine-Research/Research/Code/Model_Iterations/11:20/work/save/config.json', False,
#     test_mode=True)

# _, test_nodes, test_edges, test_targets = dataset.extract_data_create_datasets(test_model_settings)

ast_dataset = dataset.ASTDataset(train_model_settings)

'''
================================================ 

STEP 2 : MAKE DATASET ITERABLE

================================================
'''

train_batch_size = config.batch_size
num_epochs = config.num_epochs

train_nodes_loader = torch.utils.data.DataLoader(dataset=train_nodes,
                                                 batch_size=train_batch_size,
                                                 shuffle=False)
train_edges_loader = torch.utils.data.DataLoader(dataset=train_edges,
                                                 batch_size=train_batch_size,
                                                 shuffle=False)
train_targets_loader = torch.utils.data.DataLoader(dataset=train_targets,
                                                   batch_size=train_batch_size,
                                                   shuffle=False)

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

ast_data_loader = torch.utils.data.DataLoader(dataset=ast_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

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
    Encoder class
    """

    def __init__(self, config, batch_size, input_dim, hidden_units, num_layers, emb):
        super(TreeRNN, self).__init__()

        self.config = config

        self.batch_size = batch_size

        # Hidden units dimensions
        self.hidden_units = hidden_units

        # Number of hidden layers
        self.num_layers = num_layers

        # print("input dim:", input_dim)
        # GRU to handle child edges
        self.child_rnn = nn.GRU(input_dim, hidden_units, num_layers, batch_first=True)

        # GRU to handle sibling edges
        self.sibling_rnn = nn.GRU(input_dim, hidden_units, num_layers, batch_first=True)

        # # Readout layer
        # self.fc = nn.Linear(hidden_units, output_units)

        self.initial_state = torch.zeros([num_layers, batch_size, self.hidden_units])
        self.state = torch.zeros([num_layers, batch_size, self.hidden_units])

        # Embedding
        self.emb = emb

        # Outputs
        self.outputs = []

    def forward(self, nodes, edges, initial_state):
        # Create generator for node embeddings
        # print("nodes: ", nodes)
        emb_inp = (self.emb(i.to(dtype=torch.long)) for i in nodes)  # size = [batch size, max ast depth, num_units]
        # print("embedding size: ", self.emb(nodes[0].to(dtype=torch.long)).size())


        self.initial_state = initial_state
        self.state = initial_state
        self.outputs = []

        curr_out = torch.zeros([self.batch_size, self.hidden_units])
        curr_state = self.state

        for i, inp in enumerate(emb_inp):
            # print("i: ", i)
            # print("inp: ", inp.size())
            # new = inp.view(inp.size()[0], inp.size()[1], -1)
            # print("new: ", new)
            # input needs to be of size (seq_len, batch, input_dim) which should be
            # (max_ast_depth, batch_size, embedding_size) and embedding_size == num_units
            inp = inp.view(self.config.decoder.max_ast_depth, 1, self.config.decoder.units)
            output_c, state_c = self.child_rnn(inp, self.state)
            output_s, state_s = self.sibling_rnn(inp, self.state)

            # print(state_c.size())
            # print(output_c.size())
            # print(output_s.size())
            # print(state_s.size())
            # print(edges[i].size())
            # print("output type: ", type(output_c[0]))

            output = torch.where(edges[i], output_c.T, output_s.T)
            output = output.T
            # print(output_c[12])
            # print(output_s[12])
            # print(output.T[12])
            # print(output.size())
            self.outputs.append(output)
            curr_out = torch.where(torch.ne(inp, 0), output, curr_out)

            # print(state_c[0])
            test = np.array([torch.where(edges[i].view(32,1), state_c[0], state_s[0])])
            # print(test[0])
            # https://discuss.pytorch.org/t/how-to-turn-a-list-of-tensor-to-tensor/8868/5
            self.state = torch.stack([torch.where(edges[i].view(config.decoder.max_ast_depth, 1), state_c[j], state_s[j]) for j in range(self.num_layers)])
            # print(state_c[0][0])
            # print(state_s[0][0])
            # print(self.states[0])
            # print(states.size())
            curr_state = torch.stack([torch.where(torch.ne(inp, 0), self.state[j], curr_state[j]) for j in range(self.num_layers)])

        # self.last_output = self.fc(curr_out)

        return torch.stack(self.outputs), self.state

    # def init_hidden(self, state):
    #     self.initial_state = state
    #     self.state = state
    #     return


class VRNN(nn.Module):
    """
    Variational Autoencoder
    """

    def __init__(self, model_settings, config):
        super(VRNN, self).__init__()

        self.config = config

        # max length of ast
        self.max_ast_length = config.decoder.max_ast_depth

        self.latent_size = config.latent_size
        self.num_layers = config.decoder.num_layers
        # self.dec_num_layers = config.decoder.num_layers
        # self.enc_hidden_size = config.reverse_encoder.units
        self.hidden_size = config.decoder.units

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

        batch_size = config.batch_size
        # print("batch size: ", config.batch_size)
        # print("nodes size: ", nodes.size())

        # ENCODER

        # std = torch.from_numpy(np.ones([batch_size, self.hidden_size]) * float(0.001))
        enc_initial_state = torch.empty([self.num_layers, self.config.decoder.max_ast_depth, self.hidden_size]).normal_(mean=0, std=0.001)
        # self.encoder.init_hidden(enc_initial_state)

        _, hidden = self.encoder(nodes, edges, enc_initial_state)

        if self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size * self.num_layers)
        else:
            # removes all the dimensions that are 1
            hidden = hidden.squeeze()

        # hidden = hidden.view(batch_size, self.hidden_size * self.num_layers)

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        # print("hidden: ", hidden.size())
        # print("latent size: ", self.latent_size)
        # print("hidden size: ", self.hidden_size)
        # print("mean: ", mean.size())
        # print("logv: ", logv.size())
        # print("std: ", std.size())

        # z = to_var(torch.randn([batch_size, self.latent_size]))
        z = to_var(torch.randn([self.config.decoder.max_ast_depth, self.latent_size]))
        # print("z: ", z.size())
        z = z * std + mean

        # DECODER
        hidden = self.latent2hidden(z)

        if self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

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
        outputs, _ = self.decoder(nodes, edges, hidden)
        # print(outputs.size())
        # if batch_size != 1:
        outputs = outputs.squeeze()

        # # process outputs
        # padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        # padded_outputs = padded_outputs.contiguous()
        # _, reversed_idx = torch.sort(sorted_idx)
        # padded_outputs = padded_outputs[reversed_idx]
        # b, s, _ = padded_outputs.size()
        # print("decoder outputs size: ", outputs.size())
        b, s, _ = outputs.size()
        # print("b, s: ", b, s)

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(outputs.view(-1, outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        # print("logp: ", logp.size())

        return logp, mean, logv, z


'''
================================================ 

STEP 4 : INSTANTIATE MODEL CLASS

================================================
'''

# model = VRNN(train_model_settings, config)

'''
================================================ 

STEP 5 : INSTANTIATE LOSS CLASS

================================================
'''

'''
================================================ 

STEP 6 : INSTANTIATE OPTIMIZER CLASS

================================================
'''

'''
================================================ 

STEP 7 : TRAIN THE MODEL

================================================
'''


def train(model_settings, config, data_loader):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    model = VRNN(model_settings, config)

    # if torch.cuda.is_available():
    #     model = model.cuda()
    #
    # print(model)

    if model_settings.save_outputs and not model_settings.test_mode:
        os.makedirs(model_settings.output_model_path, exist_ok=True)

    # def kl_anneal_function(anneal_function, step, k, x0):
    #     if anneal_function == 'logistic':
    #         return float(1 / (1 + np.exp(-k * (step - x0))))
    #     elif anneal_function == 'linear':
    #         return min(1, step / x0)

    def kl_anneal_function(step, x0):
        return min(1, step / x0)

    # NLL = torch.nn.NLLLoss(size_average=False)
    NLL = torch.nn.NLLLoss()

    def loss_fn(logp, target, mean, logv, step, x0):

        target = target.to(dtype=torch.long)

        # cut-off unnecessary padding from target, and flatten
        target = target.view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(step, x0)

        return NLL_loss, KL_loss, KL_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    for epoch in range(config.num_epochs):

        # for split in splits:

        tracker = collections.defaultdict(tensor)


        model.train()

        for iteration, batch in enumerate(data_loader):

            batch_size = batch['node'].size(0)

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = to_var(v)

            # Forward pass
            logp, mean, logv, z = model(batch['node'], batch['edge'])

            # loss calculation
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'], mean, logv, step, 2500)

            loss = (NLL_loss + KL_weight * KL_loss) / batch_size



            # backward + optimization
            ## if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            # print("loss items: ", loss.item())

            # print("tracker: ", tracker['ELBO'])
            # print("loss.data: ", loss.data.size())
            #
            # # bookkeepeing
            # tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data))

            # if args.tensorboard_logging:
            #     writer.add_scalar("%s/ELBO" % split.upper(), loss.data[0], epoch * len(data_loader) + iteration)
            #     writer.add_scalar("%s/NLL Loss" % split.upper(), NLL_loss.data[0] / batch_size,
            #                       epoch * len(data_loader) + iteration)
            #     writer.add_scalar("%s/KL Loss" % split.upper(), KL_loss.data[0] / batch_size,
            #                       epoch * len(data_loader) + iteration)
            #     writer.add_scalar("%s/KL Weight" % split.upper(), KL_weight, epoch * len(data_loader) + iteration)

            if iteration % config.print_step == 0 or iteration + 1 == len(data_loader):
                print("Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                      % (
                      iteration, len(data_loader) - 1, loss.item(), NLL_loss.item() / batch_size,
                      KL_loss.item() / batch_size, KL_weight))

            # if split == 'valid':
            #     if 'target_sents' not in tracker:
            #         tracker['target_sents'] = list()
            #     tracker['target_sents'] += idx2word(batch['target'].data, i2w=datasets['train'].get_i2w(),
            #                                         pad_idx=datasets['train'].pad_idx)
            #     tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

        # print(
        #     "Epoch %02d/%i, Mean ELBO %9.4f" % (epoch, config.num_epochs, torch.mean(tracker['ELBO'])))

        # if args.tensorboard_logging:
        #     writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

        # # save a dump of all sentences and the encoded latent space
        # if split == 'valid':
        #     dump = {'target_sents': tracker['target_sents'], 'z': tracker['z'].tolist()}
        #     if not os.path.exists(os.path.join('dumps', ts)):
        #         os.makedirs('dumps/' + ts)
        #     with open(os.path.join('dumps/' + ts + '/valid_E%i.json' % epoch), 'w') as dump_file:
        #         json.dump(dump, dump_file)

        # save checkpoint
        # if split == 'train':
        if model_settings.save_outputs:
            torch.save(model.state_dict(), model_settings.output_model_path)
            print("Model saved at %s" % model_settings.output_model_path)


train(train_model_settings, config, ast_data_loader)