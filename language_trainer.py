import os
import argparse
import json
import sys
import copy

CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3}


def parse_args():
    """
    Returns arguments passed at the command line as a dict
    """
    parser = argparse.ArgumentParser(description='Trains a language translation model.')
    parser.add_argument('-c', help="Configuration File Location", required=True,
                        dest='config')
    arguments = vars(parser.parse_args())
    return arguments


def load_config(config_name):
    """
    loads a json config file and returns a config dictionary
    """
    with open(config_name) as config_file:
        configuration = json.load(config_file)
        return configuration


def read_file(file_path):
    """
    Reads file in for language translation.
    :param file_path: File path for either source or target data.
    :return: opened file object
    """
    input_file = os.path.join(file_path)
    with open(input_file, 'r', encoding='utf-8') as f:
        return f.read()


def lower_text(text):
    """
    Converts a text string to lowercase.
    :param text: Input text string
    :return: text converted to lowercase
    """
    return text.lower()


def create_lookup_tables(text):
    """
    Creates lookup tables for vocabulary words and indexes.
    :param text: Input text
    :return: vocab-to-int, and int-to-vocab dictionaries
    """
    vocab = set(text.split())
    vocab_to_int = copy.copy(CODES)

    for vocab_index, vocab in enumerate(vocab, len(CODES)):
        vocab_to_int[vocab] = vocab_index

    int_to_vocab = {vocab_index: vocab for vocab, vocab_index in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


if __name__ == '__main__':

    args = parse_args()
    config = load_config(args['config'])

    # PREPROCESSING

    # Load the datasets
    x = read_file(config['x_language_path'])
    y = read_file(config['y_language_path'])

    # Convert to lowercase
    x = lower_text(x)
    y = lower_text(y)

    # Create lookup dictionaries
    x_vocab_to_int, x_int_to_vocab = create_lookup_tables(x)
    y_vocab_to_int, y_int_to_vocab = create_lookup_tables(y)

    sys.exit()