import os
import argparse
import json
import sys


def parse_args():
    """
    Returns arguments passed at the command line as a dict
    """
    parser = argparse.ArgumentParser(description='Trains a language translation model.')
    parser.add_argument('-c', help="Configuration File Location", required=True,
                        dest='config')
    args = vars(parser.parse_args())
    return args


def load_config(config_name):
    """
    loads a json config file and returns a config dictionary
    """
    with open(config_name) as config_file:
        config = json.load(config_file)
        return config


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


if __name__ == '__main__':

    args = parse_args()
    config = load_config(args['config'])

    # Load the datasets
    x = read_file(config['x_language_path'])
    y = read_file(config['y_language_path'])

    sys.exit()




