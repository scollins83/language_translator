import tensorflow as tf
import pickle
import argparse
import sys


def parse_args(arguments):
    """

    :param arguments:
    :return:
    """
    parser = argparse.ArgumentParser(description="Translates a sentence from English to French.")
    parser.add_argument('-i', '--input', help='Input sentence--- should be in English.', required=True,
                        dest='input_sentence')
    return vars(parser.parse_args(arguments))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    with open('201709052013_French_to_English/preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)


def load_params():
    """
    Load parameters from file
    """
    with open('201709052013_French_to_English/params.p', mode='rb') as in_file:
        return pickle.load(in_file)


def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    sentence = sentence.lower()
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in sentence.split(' ')]


def translate(translate_sentence, batch_size=256):
    """

    :param translate_sentence:
    :param batch_size:
    :return:
    """
    _, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = load_preprocess()
    load_path = load_params()

    translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        translate_logits = sess.run(logits, {input_data: [translate_sentence] * batch_size,
                                             target_sequence_length: [len(translate_sentence) * 2] * batch_size,
                                             source_sequence_length: [len(translate_sentence)] * batch_size,
                                             keep_prob: 1.0})[0]

        return " ".join([target_int_to_vocab[i] for i in translate_logits])


if __name__ == '__main__':

    args = parse_args(sys.argv[1:])
    input_sentence = args['input_sentence']

    print(translate(input_sentence))

    sys.exit()
