import unittest
from itertools import product

alphabet_value = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x', 'y', 'z']


class FeatureManager:
    def __init__(self, len_of_key, alphabet_key, len_of_key_back=None, alphabet_key_back=None):
        """

        :param len_of_key: do dai cua chuoi feature dau tien
        :param alphabet_key: tap cac cach chon cua feature 1
        :param len_of_key_back: do dai cua chuoi feature thu hai, noi voi chuoi dau
        :param alphabet_key_back: tap cac cach chon cua feature 2
        """
        self.len_of_key_back = len_of_key_back
        self.alphabet_key_back = alphabet_key_back
        self.len_of_key = len_of_key
        self.alphabet_key = alphabet_key
        self.alphabet_value = alphabet_value
        self.num_of_key = len(alphabet_key) ** len_of_key

    def get_dictionary(self, is_merge_list=False):
        if is_merge_list:
            list_key = self.get_list_key_merge()
            len_key = len(list_key)
            list_value = self.get_list_value(len_key)
        else:
            list_key = self.get_list_key()
            list_value = self.get_list_value()

        dictionary = dict(zip(list_key, list_value))
        return dictionary

    def get_list_key(self, is_back=False):
        if not is_back:
            list_key_separate = product(self.alphabet_key, repeat=self.len_of_key)
            list_key = ["".join(i) for i in list(list_key_separate)]
            return list_key
        else:
            if self.alphabet_key_back is None:
                raise ValueError("Alphabet of list key is missing")
            list_key_separate = product(self.alphabet_key_back, repeat=self.len_of_key_back)
            list_key = ["".join(i) for i in list(list_key_separate)]
            return list_key

    def get_list_key_merge(self):
        list_key_front = self.get_list_key()
        list_key_back = self.get_list_key(is_back=True)
        out = []
        for front in list_key_front:
            for back in list_key_back:
                out.append(front + back)
        return out

    def get_list_value(self, num_of_key_merge=0):

        len_space = len(self.alphabet_value)
        len_value = 0

        if num_of_key_merge == 0:
            num_of_key = self.num_of_key
        else:
            num_of_key = num_of_key_merge

        while (len_space ** len_value < num_of_key):
            len_value += 1

        list_of_value_full_separate = list(product(self.alphabet_value, repeat=len_value))
        list_of_value_full = ["".join(i) for i in list(list_of_value_full_separate)]
        list_of_value = list_of_value_full[:num_of_key]
        return list_of_value


class TestFeatureManager(unittest.TestCase):

    def setUp(self):
        alphabet_key = ['L', 'S', 'N']
        alphabet_key_back = ['a', 'b']
        self.feature = FeatureManager(3, alphabet_key, 2, alphabet_key_back)

    def test_get_list_key(self):
        self.assertEqual(len(self.feature.get_list_key()), self.feature.num_of_key)

    def test_get_list_value(self):
        print self.feature.get_list_value()

    def test_get_list_key_merge(self):
        list_key_merge = self.feature.get_list_key_merge()
        print list_key_merge

    def test_get_dictionary(self):
        print self.feature.get_dictionary()

    def test_get_dictionary_merge(self):
        print self.feature.get_dictionary(is_merge_list=True)


if __name__ == '__main__':
    unittest.main()
