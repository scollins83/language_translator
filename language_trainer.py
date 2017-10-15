import os
import argparse
import json
import sys
import copy
import pickle
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np

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


def model_inputs():
    """
    Establish model inputs for tensorflow graph.
    :return:Tuple (input, targets, learning rate, keep probability, target sequence length,
    max target sequence length, source sequence length)
    """
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    keep_probability = tf.placeholder(tf.float32, shape=(), name='keep_prob')
    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, [None], name='source_sequence_length')
    return inputs, targets, learning_rate, keep_probability, target_sequence_length, max_target_len, source_sequence_length


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding.
    :param target_data:
    :param target_vocab_to_int:
    :param batch_size:
    :return:
    """
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    preprocessed_target_data = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), ending], 1)

    return preprocessed_target_data


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,
                   source_sequence_length, source_vocab_size,
                   encoding_embedding_size):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :param source_sequence_length: a list of the lengths of each sequence in the batch
    :param source_vocab_size: vocabulary size of source data
    :param encoding_embedding_size: embedding size of source data
    :return: tuple (RNN output, RNN state)
    """
    # Encoder embedding
    encoder_embed_input = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoding_embedding_size)

    # RNN Cell
    def make_cell(rnn_size):
        encoder_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=73))
        drop = tf.contrib.rnn.DropoutWrapper(encoder_cell, output_keep_prob=keep_prob)
        return drop

    encoder_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])

    encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_embed_input,
                                                      sequence_length=source_sequence_length, dtype=tf.float32)

    return encoder_output, encoder_state


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         target_sequence_length, max_summary_length,
                         output_layer, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_summary_length: The length of the longest sequence in the batch
    :param output_layer: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input, sequence_length=target_sequence_length,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, training_helper, encoder_state, output_layer)

    training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                                maximum_iterations=max_summary_length)[0]

    return training_decoder_output


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param max_target_sequence_length: Maximum length of target sequences
    :param vocab_size: Size of decoder/target vocabulary
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_layer: Function to apply the output layer
    :param batch_size: Batch size
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [batch_size], name='start_tokens')

    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens, end_of_sequence_id)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, encoder_state, output_layer)

    inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder, impute_finished=True,
                                                                 maximum_iterations=max_target_sequence_length)[0]

    return inference_decoder_output


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    # 1. Decoder embedding
    # target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    # 2. Construct the decoder cell
    def make_cell(rnn_size):
        dec_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=73))
        return dec_cell

    dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])

    # 3. Dense layer to translate decoder's output
    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    # 4. Set up a training decoder and an inference decoder.
    # Training Decoder
    with tf.variable_scope("decode"):
        # Perform dynamic decoding using the decoder
        train_decoder_output = decoding_layer_train(encoder_state,
                                                    dec_cell,
                                                    dec_embed_input,
                                                    target_sequence_length,
                                                    max_target_sequence_length,
                                                    output_layer,
                                                    keep_prob)

    # 5. Inference Decoder
    # Reuses the same parameters trained by the training process
    with tf.variable_scope("decode", reuse=True):
        # User inference
        inference_decoder_output = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings,
                                                        target_vocab_to_int['<GO>'],
                                                        target_vocab_to_int['<EOS>'],
                                                        max_target_sequence_length, target_vocab_size, output_layer,
                                                        batch_size, keep_prob)

    return train_decoder_output, inference_decoder_output


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  source_sequence_length, target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param source_sequence_length: Sequence Lengths of source sequences in the batch
    :param target_sequence_length: Sequence Lengths of target sequences in the batch
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    _, enc_state = encoding_layer(input_data,
                                  rnn_size,
                                  num_layers,
                                  keep_prob,
                                  source_sequence_length,
                                  source_vocab_size,
                                  enc_embedding_size)

    dec_input = process_decoder_input(target_data, target_vocab_to_int,
                                      batch_size)

    # Pass encoder state and decoder inputs to the decoders
    training_decoder_output, inference_decoder_output = decoding_layer(
        dec_input, enc_state, target_sequence_length, max_target_sentence_length,
        rnn_size, num_layers, target_vocab_to_int, target_vocab_size,
        batch_size, keep_prob, dec_embedding_size)

    return training_decoder_output, inference_decoder_output


def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths


def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0, 0), (0, max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0, 0), (0, max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))


def save_params(params, params_save_path):
    """
    Save parameters to file
    """
    with open('params.p', 'wb') as out_file:
        pickle.dump(params, out_file)


def load_params():
    """
    Load parameters from file
    """
    with open('params.p', mode='rb') as in_file:
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

    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.1'), 'Please use TensorFlow version 1.1 or newer'
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    if config['train_model'] == "True":

        max_target_sentence_length = max([len(sentence) for sentence in x_ids])

        train_graph = tf.Graph()
        with train_graph.as_default():
            input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length = model_inputs()

            # sequence_length = tf.placeholder_with_default(max_target_sentence_length, None, name='sequence_length')
            input_shape = tf.shape(input_data)

            train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                           targets,
                                                           keep_prob,
                                                           config['batch_size'],
                                                           source_sequence_length,
                                                           target_sequence_length,
                                                           max_target_sequence_length,
                                                           len(x_vocab_to_int),
                                                           len(y_vocab_to_int),
                                                           config['encoding_embedding_size'],
                                                           config['decoding_embedding_size'],
                                                           config['rnn_size'],
                                                           config['num_layers'],
                                                           y_vocab_to_int)

            training_logits = tf.identity(train_logits.rnn_output, name='logits')
            inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

            masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

            with tf.name_scope("optimization"):
                # Loss function
                cost = tf.contrib.seq2seq.sequence_loss(
                    training_logits,
                    targets,
                    masks)

                # Optimizer
                optimizer = tf.train.AdamOptimizer(lr)

                # Gradient Clipping
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if
                                    grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)

        # Split data to training and validation sets
        train_source = x_ids[config['batch_size']:]
        train_target = y_ids[config['batch_size']:]
        valid_source = x_ids[:config['batch_size']]
        valid_target = y_ids[:config['batch_size']]
        (valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths) = next(
            get_batches(valid_source,
                        valid_target,
                        config['batch_size'],
                        x_vocab_to_int['<PAD>'],
                        y_vocab_to_int['<PAD>']))
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(config['epochs']):
                for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                        get_batches(train_source, train_target, config['batch_size'],
                                    x_vocab_to_int['<PAD>'],
                                    y_vocab_to_int['<PAD>'])):

                    _, loss = sess.run(
                        [train_op, cost],
                        {input_data: source_batch,
                         targets: target_batch,
                         lr: config['learning_rate'],
                         target_sequence_length: targets_lengths,
                         source_sequence_length: sources_lengths,
                         keep_prob: config['keep_probability']})

                    if batch_i % config['display_step'] == 0 and batch_i > 0:
                        batch_train_logits = sess.run(
                            inference_logits,
                            {input_data: source_batch,
                             source_sequence_length: sources_lengths,
                             target_sequence_length: targets_lengths,
                             keep_prob: 1.0})

                        batch_valid_logits = sess.run(
                            inference_logits,
                            {input_data: valid_sources_batch,
                             source_sequence_length: valid_sources_lengths,
                             target_sequence_length: valid_targets_lengths,
                             keep_prob: 1.0})

                        train_acc = get_accuracy(target_batch, batch_train_logits)

                        valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                        print(
                            'Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                                .format(epoch_i, batch_i, len(x_ids) // config['batch_size'], train_acc, valid_acc,
                                        loss))

            # Save Model
            saver = tf.train.Saver()
            saver.save(sess, config['save_path'])
            print('Model Trained and Saved')

        save_params(config['save_path'])

    _, (source_vocab_to_int, target_vocab_to_int), (
    source_int_to_vocab, target_int_to_vocab) = load_preprocessing_data()
    load_path = load_params()

    translate_sentence = sentence_to_seq(config['translate_sentence'], source_vocab_to_int)

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

        translate_logits = sess.run(logits, {input_data: [translate_sentence] * config['batch_size'],
                                             target_sequence_length: [len(translate_sentence) * 2] * config[
                                                 'batch_size'],
                                             source_sequence_length: [len(translate_sentence)] * config['batch_size'],
                                             keep_prob: 1.0})[0]

    print('Input')
    print('  Word Ids:      {}'.format([i for i in translate_sentence]))
    print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

    print('\nPrediction')
    print('  Word Ids:      {}'.format([i for i in translate_logits]))
    print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits])))

    sys.exit()