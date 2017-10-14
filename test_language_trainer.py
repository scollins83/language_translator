import unittest
import language_trainer as lt


class MyTestCase(unittest.TestCase):

    def test_read_files(self):
        """
        Assumes two files for a parallel translation dataset.
        Checks to see if either of the incoming files is empty.
        """
        x_language_path = "data/small_vocab_en"
        y_language_path = "data/small_vocab_fr"
        x = lt.read_file(x_language_path).split('\n')
        y = lt.read_file(y_language_path).split('\n')
        self.assertEqual(len(x), len(y))

if __name__ == '__main__':
    unittest.main()
