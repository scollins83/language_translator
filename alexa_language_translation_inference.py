import logging
import tensorflow as tf
import pickle
from flask import Flask, render_template
from flask_ask import Ask, statement, question, session

app = Flask(__name__)
ask = Ask(app, "/")
logging.getLogger("flask_ask").setLevel(logging.DEBUG)

@ask.launch
def new_translation():
    welcome_msg = render_template('welcome')
    return question(welcome_msg)


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


@ask.intent("YesIntent")
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


        french_statement =  " ".join([target_int_to_vocab[i] for i in translate_logits])
        french_msg = render_template('translation', french_stmt=french_statement)
        session.attributes['french_stmt'] = french_statement
        return statement(french_msg)


if __name__ == '__main__':
    app.run(debug=True)
