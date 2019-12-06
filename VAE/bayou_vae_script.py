import argparse
import sys
import os

import bayou_vae_functions
from bayou_vae_utils import ModelSettings

PYTHON_RECURSION_LIMIT = 10000

TINY_TRAINING_DATASET_PATH = 'data/data.json'
TINY_TRAINING_DATASET_CONFIG_PATH = 'config/config.json'
LARGE_TRAINING_DATASET_PATH = 'data/DATA-extracted-for-CACM-train.json'
LARGE_TRAINING_DATASET_CONFIG_PATH = 'config/large_dataset_config.json'
TEST_DATA_PATH = 'data/test_data/'

def train_vae(data_path, config_path, save_outputs=True, test_mode=False):
    """
    Trains VAE
    :param test_mode: (boolean) saves extra information and doesn't config models
    :param data_path: (string) path to training dataset json file
    :param config_path: (string) path to config json file
    :param save_outputs: (boolean) whether or not to config outputs like model checkpoints and metadata, plots,
           and analysis file
    :return: None
    """
    model_settings = ModelSettings(data_path, config_path, save_outputs, test_mode=test_mode)
    bayou_vae_functions.train(model_settings)
    return


def train_vae_tiny_dataset(save_outputs=True, test_mode=False):
    """
    Train VAE on tiny dataset
    :param test_mode: (boolean) saves extra information and doesn't config models
    :param save_outputs: (boolean) whether or not to config outputs like model, plots, and analysis file
    :return: None
    """
    train_vae(TINY_TRAINING_DATASET_PATH, TINY_TRAINING_DATASET_CONFIG_PATH, save_outputs, test_mode)
    return


def train_vae_large_dataset(save_outputs=True, test_mode=False):
    """
    Train VAE on large dataset
    :param test_mode: (boolean) saves extra information and doesn't config models
    :param save_outputs: (boolean) whether or not to config outputs like model, plots, and analysis file
    :return: None
    """
    train_vae(LARGE_TRAINING_DATASET_PATH, LARGE_TRAINING_DATASET_CONFIG_PATH, save_outputs, test_mode)
    return


def test_vae(data_path, config_path, saved_model_path, use_beam_search=True,  save_outputs=True, test_mode=False):
    """

    :param use_beam_search: (boolean) whether to use beam search or random search when traversing
           latent space to create ast
    :param saved_model_path: path to directory where model checkpoints are
    :param data_path: path to test data json file
    :param config_path: path to config json file
    :param save_outputs: (boolean) whether or not to config outputs like model, plots, and analysis file
    :param test_mode: (boolean) saves extra information and doesn't config models
    :return: None
    """
    model_settings = ModelSettings(data_path, config_path, save_outputs, test_mode=test_mode, saved_model_path=saved_model_path)
    bayou_vae_functions.test(model_settings, use_beam_search)


def test_vae_single_program(program_number, config_path, saved_model_path, use_beam_search=True,
                            save_outputs=False, test_mode=True):
    """

    :param use_beam_search: (boolean) whether to use beam search or random search when traversing
           latent space to create ast
    :param program_number: (natural number) all sample programs have a number in their filename
           (sample_program<number>.json)
    :param config_path: (string) path to config json file
    :param saved_model_path: (string) path to directory where model checkpoints are
    :param save_outputs: (boolean) whether or not to config outputs like model, plots, and analysis file
    :param test_mode: (boolean) saves extra information and doesn't config models
    :return: None
    """
    sample_program = 'sample_program' + str(program_number) + '.json'
    data_path = os.path.join(TEST_DATA_PATH, sample_program)

    # TODO: uncomment when you actually implement this
    # test_vae(data_path, config_path, saved_model_path, save_outputs, test_mode)

    model_settings = ModelSettings(data_path, config_path, save_outputs, saved_model_path=saved_model_path, test_mode=test_mode)
    bayou_vae_functions.sample_test(model_settings, use_beam_search)


def plot_vae(data_path, config_path, saved_model_path, save_outputs=True, test_mode=False):
    model_settings = ModelSettings(data_path, config_path, save_outputs, test_mode=test_mode, saved_model_path=saved_model_path)
    bayou_vae_functions.plot(model_settings)

sys.setrecursionlimit(PYTHON_RECURSION_LIMIT)

# train_vae_tiny_dataset(test_mode=True)
# plot_vae(TINY_TRAINING_DATASET_PATH, TINY_TRAINING_DATASET_CONFIG_PATH,  '/Users/meghanachilukuri/Documents/GitHub/Jermaine-Research/Research/Code/Model Iterations/11:20/work', test_mode=True)


# plot_vae(LARGE_TRAINING_DATASET_PATH, LARGE_TRAINING_DATASET_CONFIG_PATH, '/Users/meghanachilukuri/Documents/GitHub/Jermaine-Research/Research/Code/Model Iterations/11:21/nov13')

