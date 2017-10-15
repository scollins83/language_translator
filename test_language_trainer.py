import unittest
import language_trainer as lt


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
        self.assertGreater(len(x),0, "File lengths not greater than zero.")

    def test_lowercase(self):
        """
        Function tests if lowercase.
        """
        text = "The qUick brown FoX jumped OveR the lAzy dog"
        lower_x = lt.lower_text(text)
        print(lower_x)
        lower_values = [ord(letter) for letter in lower_x if ord(letter) != 32]
        print(lower_values)
        min_value = min(lower_values)
        max_value = max(lower_values)
        self.assertGreaterEqual(min_value, 97)
        self.assertLessEqual(max_value, 122)



if __name__ == '__main__':
    unittest.main()