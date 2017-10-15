import os
import argparse
import json
import sys
import copy
import pickle

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


def text_to_ids(text_list, vocab_to_int):
    """
    Converts text to vocabulary integers.
    :param text_list: Input list of text items.
    :param vocab_to_int: Vocab-to-int dictionary.
    :return: List of integer codes for each word.
    """
    text_ids = [[vocab_to_int.get(word, vocab_to_int['<UNK>'])
                 for word in entry.split(' ')] for entry
                in str(text_list).split('\n')]

    return text_ids


def add_eos_tag_to_ids(id_text_list, eos_tag):
    """
    Adds 'eos' (end-of-sentence) tag to end of each sentence.
    :param id_text_list: List of id-converted sentences.
    :param eos_tag: EOS tag value.
    :return: id_text_list with eos_tag value added to the end of each sentence.
    """
    for sentence in id_text_list:
        sentence.append(eos_tag)

    return id_text_list


def save_preprocessing_data(x_id_sequence, y_id_sequence,
                           x_vocab_to_int, y_vocab_to_int,
                           x_int_to_vocab, y_int_to_vocab,
                           preprocess_path):
    """
    Saves the x and y parameter objects to a pickle file.
    :param x_id_sequence:
    :param y_id_sequence:
    :param x_vocab_to_int:
    :param y_vocab_to_int:
    :param x_int_to_vocab:
    :param y_int_to_vocab:
    :param preprocess_path: Path to save the pickle to.
    :return: Saves the object as a pickle.
    """
    with open(preprocess_path, 'wb') as out_file:
        pickle.dump((
            (x_id_sequence, y_id_sequence),
            (x_vocab_to_int, y_vocab_to_int),
            (x_int_to_vocab, y_int_to_vocab)
        ), out_file)

    return 1


def load_preprocessing_data(preprocess_path):
    """
    Loads preprocessing data from existing pickle file.
    :param preprocess_path: Path of the pre-existing preprocess pickle.
    :return: x_ids, y_ids, x_vocab-to-int, y_vocab-to-int
    """
    with open(preprocess_path, 'rb') as in_file:
        return pickle.load(in_file)


if __name__ == '__main__':

    args = parse_args()
    config = load_config(args['config'])

    # PREPROCESSING

    if config["use_existing_preprocessing"] == "False":

        # Load the datasets
        x = read_file(config['x_language_path'])
        y = read_file(config['y_language_path'])

        # Convert to lowercase
        x = lower_text(x)
        y = lower_text(y)

        # Create lookup dictionaries
        x_vocab_to_int, x_int_to_vocab = create_lookup_tables(x)
        y_vocab_to_int, y_int_to_vocab = create_lookup_tables(y)

        # Convert text to ids
        x_ids = text_to_ids(x, x_vocab_to_int)
        y_ids = text_to_ids(y, y_vocab_to_int)
        y_ids = add_eos_tag_to_ids(y_ids, y_vocab_to_int['<EOS>'])

        # Save the preprocessing objects.
        save_preprocessing_data(x_ids, y_ids, x_vocab_to_int, y_vocab_to_int,
                                x_int_to_vocab, y_int_to_vocab,
                                config['preprocessing_path'])

    # CONSTRUCT NEURAL NETWORK
    (x_ids, y_ids), (x_vocab_to_int,
                     y_vocab_to_int), _ = load_preprocessing_data(config['preprocessing_path'])




    sys.exit()