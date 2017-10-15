import unittest
import language_trainer as lt
import glob
import os
import tensorflow as tf


class MyTestCase(unittest.TestCase):

    def test_read_files(self):
        """
        Assumes two files for a parallel translation dataset.
        Checks to see if the lengths of the incoming files are equal,
        and then checks to see that they're greater than zero.
        """
        x_language_path = "data/small_vocab_en"
        y_language_path = "data/small_vocab_fr"
        x = lt.read_file(x_language_path).split('\n')
        y = lt.read_file(y_language_path).split('\n')
        self.assertEqual(len(x), len(y), "File lengths unequal.")
        self.assertGreater(len(x), 0, "File lengths not greater than zero.")

    def test_lowercase(self):
        """
        Function tests if lowercase.
        """
        text = "The qUick brown FoX jumped OveR the lAzy dog"
        lower_x = lt.lower_text(text)
        lower_values = [ord(letter) for letter in lower_x if ord(letter) != 32]
        min_value = min(lower_values)
        max_value = max(lower_values)
        self.assertGreaterEqual(min_value, 97)
        self.assertLessEqual(max_value, 122)

    def test_create_lookup_tables(self):
        """
        Creates vocab-to-int and int-to-vocab dictionaries.
        """
        text = "the quick brown fox jumped over the lazy dog"
        vocab_to_int, int_to_vocab = lt.create_lookup_tables(text)
        self.assertEqual(len(vocab_to_int), len(int_to_vocab), "Lookup dictionaries \
                         are not the same length.")
        self.assertEqual(vocab_to_int['<EOS>'], 1, "Codes not added to index.")

    def test_text_to_ids(self):
        """
        Converts text strings to integer ids using the vocab-to-int
         dictionary as a lookup table.
        """
        text = "the quick brown fox jumped over the lazy dog\n now he jumped over the fence"
        vocab_to_int, int_to_vocab = lt.create_lookup_tables(text)
        id_sequence = lt.text_to_ids(text, vocab_to_int)
        self.assertEqual(len(text.split('\n')[0].split()), len(id_sequence[0]), "Length changed.")

    def test_add_eos_tag_to_ids(self):
        """
        Adds end-of-sentence (eos) tag to the end of each sentence supplied.
        """
        text = "the quick brown fox jumped over the lazy dog\n now he jumped over the fence"
        vocab_to_int, int_to_vocab = lt.create_lookup_tables(text)
        id_sequence = lt.text_to_ids(text, vocab_to_int)
        eos_id_sequence = lt.add_eos_tag_to_ids(id_sequence, vocab_to_int['<EOS>'])
        self.assertEqual(eos_id_sequence[0][-1], 1)
        self.assertEqual(eos_id_sequence[1][-1], 1)

    def test_save_preprocessing_data(self):
        """
        Saves preprocessing objects together in one file.
        """
        preprocess_path = "test_objects/test_preprocess.p"
        text = "the quick brown fox jumped over the lazy dog\n now he jumped over the fence"
        text2 = "the quick brown fox jumped over the lazy dog\n now he jumped over the fence"
        x_vocab_to_int, x_int_to_vocab = lt.create_lookup_tables(text)
        y_vocab_to_int, y_int_to_vocab = lt.create_lookup_tables(text2)
        x_id_sequence = lt.text_to_ids(text, x_vocab_to_int)
        y_id_sequence = lt.text_to_ids(text2, y_vocab_to_int)
        y_id_sequence = lt.add_eos_tag_to_ids(y_id_sequence, y_vocab_to_int['<EOS>'])
        lt.save_preprocessing_data(x_id_sequence, y_id_sequence,
                                   x_vocab_to_int, y_vocab_to_int,
                                   x_int_to_vocab, y_int_to_vocab,
                                   preprocess_path)
        file_list = glob.glob("test_objects/*")
        self.assertEqual(file_list[0], "test_objects/test_preprocess.p")
        os.remove(preprocess_path)

    def test_load_preprocessing_data(self):
        """
        k
        """
        preprocess_path = "test_objects/test_preprocess.p"
        text = "the quick brown fox jumped over the lazy dog\n now he jumped over the fence"
        text2 = "the quick brown fox jumped over the lazy dog\n now he jumped over the fence"
        x_vocab_to_int, x_int_to_vocab = lt.create_lookup_tables(text)
        y_vocab_to_int, y_int_to_vocab = lt.create_lookup_tables(text2)
        x_id_sequence = lt.text_to_ids(text, x_vocab_to_int)
        y_id_sequence = lt.text_to_ids(text2, y_vocab_to_int)
        y_id_sequence = lt.add_eos_tag_to_ids(y_id_sequence, y_vocab_to_int['<EOS>'])
        lt.save_preprocessing_data(x_id_sequence, y_id_sequence,
                                   x_vocab_to_int, y_vocab_to_int,
                                   x_int_to_vocab, y_int_to_vocab,
                                   preprocess_path)
        (x_id_sequence_load, y_id_sequence_load), \
        (x_vocab_to_int_load, y_vocab_to_int_load), _ = lt.load_preprocessing_data(preprocess_path)
        self.assertEqual(x_id_sequence, x_id_sequence_load)
        self.assertEqual(y_id_sequence, y_id_sequence_load)
        self.assertEqual(x_vocab_to_int, x_vocab_to_int_load)
        self.assertEqual(y_vocab_to_int, y_vocab_to_int_load)
        os.remove(preprocess_path)

    def test_model_inputs(self):
        """
        Generates input placeholders for the tensorflow graph.
        """
        with tf.Graph().as_default():
            input_data, targets, learning_rate, keep_prob, \
            target_sequence_length, \
            max_target_sequence_length, \
            source_sequence_length = lt.model_inputs()

        self.assertEqual(input_data.op.type, 'Placeholder', 'Input is not a placeholder.')
        self.assertEqual(targets.op.type, 'Placeholder', 'Targets are not a placeholder.')
        self.assertEqual(learning_rate.op.type, 'Placeholder', 'Learning rate is not a placeholder.')
        self.assertEqual(keep_prob.op.type, 'Placeholder', 'Keep probability for dropout is not a placeholder.')
        self.assertEqual(target_sequence_length.op.type, 'Placeholder',
                         'Target sequence length is not a placeholder.')
        self.assertEqual(max_target_sequence_length.op.type,
                         'Max',
                         'Max target sequence length is not a Max.')
        self.assertEqual(input_data.op.type, 'Placeholder', 'Input is not a placeholder.')

        self.assertEqual(input_data.name,'input:0',
                         'Input has bad name.  Found name {}'.format(input_data.name))
        self.assertEqual(target_sequence_length.name,'target_sequence_length:0',
                         'Target Sequence Length has bad name.  Found name {}'.format(
                             target_sequence_length.name))
        self.assertEqual(source_sequence_length.name,
                         'source_sequence_length:0','Source Sequence Length has bad name. Found name {}'.format(source_sequence_length.name))
        self.assertEqual(keep_prob.name, 'keep_prob:0', 'Keep Probability has bad name.  Found name {}'.format(keep_prob.name))

        assert tf.assert_rank(input_data, 2, message='Input data has wrong rank')
        assert tf.assert_rank(targets, 2, message='Targets has wrong rank')
        assert tf.assert_rank(learning_rate, 0, message='Learning Rate has wrong rank')
        assert tf.assert_rank(keep_prob, 0, message='Keep Probability has wrong rank')
        assert tf.assert_rank(target_sequence_length, 1, message='Target Sequence Length has wrong rank')
        assert tf.assert_rank(max_target_sequence_length, 0, message='Max Target Sequence Length has wrong rank')
        assert tf.assert_rank(source_sequence_length, 1, message='Source Sequence Length has wrong rank')



if __name__ == '__main__':
    unittest.main()
