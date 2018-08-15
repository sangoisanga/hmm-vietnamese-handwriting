import unittest
from itertools import product

alphabet_value = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x', 'y', 'z']


class FeatureManager:
    def __init__(self, len_of_key, alphabet_key, list_other=tuple(), list_component=None):
        """
        :param len_of_key: do dai cua chuoi feature dau tien
        :param alphabet_key: tap cac cach chon cua feature 1
        :param len_of_key_back: do dai cua chuoi feature thu hai, noi voi chuoi dau
        :param alphabet_key_back: tap cac cach chon cua feature 2
        """
        self.list_component = list_component
        self.num_of_other_feature = len(list_other)
        self.list_other = list_other
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
            if self.num_of_other_feature == 0:
                raise ValueError("List other feature must not empty!")
            else:
                list_of_list_key = []
                for length, alphabet in self.list_other:
                    list_key_separate = product(alphabet, repeat=length)
                    list_key = ["".join(i) for i in list(list_key_separate)]
                    list_of_list_key.append(list_key)
            return list_of_list_key

    def get_list_key_merge(self):
        list_key_front = self.get_list_key()
        list_key_back = self.get_list_key(is_back=True)

        start_function = "product(list_key_front"
        end_function = ",repeat=1)"

        for i in range(len(list_key_back)):
            start_function += ",list_key_back[" + str(i) + "]"
        if self.list_component is not None:
            start_function += ",self.list_component"
        #print start_function + end_function
        out = list(eval(start_function + end_function))
        list_key = ["".join(i) for i in out]
        return list_key

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
        alphabet_key = ['L', 'S']
        alphabet_key_back = ['a', 'b']
        alphabet_key_back2 = ['P', 'Q']

        self.feature = FeatureManager(3, alphabet_key, [(2, alphabet_key_back), (2, alphabet_key_back2)],
                                      ["ABC", "BAC"])

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
        a = self.feature.get_dictionary(is_merge_list=True)
        print a
        print [v for k, v in a.items()]


if __name__ == '__main__':
    unittest.main()
