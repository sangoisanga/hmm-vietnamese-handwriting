import glob
import os
import unittest
import math
from random import random

import cv2

from main.feature.image_example_dir import ImageExampleDir
from main.feature.image_preprocessor import scale_to_fill, divide_into_segments, extract_sorted_component_size_list, extract_upper_contour


class SimpleImageFeatureExtractor(object):
    '''
    A class used to extract a sequence of features from an image that
    may be used as training observations for a HMM.
    '''
    feature_ids = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    feature_upper_contour_ids = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
        'u', 'v', 'w', 'x', 'y', 'z', '1', '2', '3', '4',
        '5', '6',
    ]
    feature_pattern_to_id = {"LLL": "a",
                             "LLS": "b",
                             "LSS": "c",
                             "LSN": "d",
                             "LLN": "e",
                             "LNN": "f",
                             "SSS": "g",
                             "SSN": "h",
                             "SNN": "i",
                             "NNN": "j"}
    feature_upper_contour_pattern_to_id = {
        'LLLLL':'a', 'LLLLS':'b', 'LLLSL':'c', 'LLLSS':'d', 'LLSLL':'e', 'LLSLS':'f', 'LLSSL':'g',
        'LLSSS':'h', 'LSLLL':'i', 'LSLLS':'j', 'LSLSL':'k', 'LSLSS':'l', 'LSSLL':'m', 'LSSLS':'n',
        'LSSSL':'o', 'LSSSS':'p', 'SLLLL':'q', 'SLLLS':'r', 'SLLSL':'s', 'SLLSS':'t', 'SLSLL':'u',
        'SLSLS':'v', 'SLSSL':'w', 'SLSSS':'x', 'SSLLL':'y', 'SSLLS':'z', 'SSLSL':'1', 'SSLSS':'2',
        'SSSLL':'3', 'SSSLS':'4', 'SSSSL':'5', 'SSSSS':'6'
    }

    def __init__(self,
                 nr_of_divisions=7, upper_contour_divisions = 5,
                 size_classification_factor=1.3, upper_contour_threshold = 0.5
                 ):
        '''
        Parameters:
        * nr_of_divisions - Number of times to divide the image vertically
        * size_classification_factor -  A component in a segment is classified
        as small if the component size is less than "segment_width * size_classification_factor"
        and greater than zero otherwise it is classified as large. Zero size segments are
        classified as none.
        * nr_of_components_to_consider - The number of components to consider

        The 3 largest components in a segment are used to get a feature for that segment.
        There are 10 different possible features in every segment. The features are enumerated
        in the following list:

        feature id | comp. 1 | comp. 2 | comp. 3
        a          | L       | L       | L       |
        b          | L       | L       | S       |
        c          | L       | S       | S       |
        d          | L       | S       | N       |
        e          | L       | L       | N       |
        f          | L       | N       | N       |
        g          | S       | S       | S       |
        h          | S       | S       | N       |
        i          | S       | N       | N       |
        j          | N       | N       | N       |

        comp. = component
        L = large
        S = small
        N = none
        '''
        self.nr_of_divisions = nr_of_divisions
        self.size_classification_factor = size_classification_factor
        self.upper_contour_divisions = upper_contour_divisions
        self.upper_contour_threshold = upper_contour_threshold

    def extract_feature_string(self, buffered_image):
        scaled_image = scale_to_fill(buffered_image)
        segments = divide_into_segments(self.nr_of_divisions, scaled_image)
        # Get component sizes for the segments
        features_for_segments = [extract_sorted_component_size_list(s)
                                 for s in segments]

        # Make sure that there are 3 elements on the list for all segmensts
        def make_size_of_list3(list):
            if len(list) == 3:
                return list
            elif len(list) > 3:
                del list[len(list) - 1]
                return make_size_of_list3(list)
            elif len(list) < 3:
                list.append(0)
                return make_size_of_list3(list)

        features_for_segments = [make_size_of_list3(l)
                                 for l in features_for_segments]

        def classify_component(component_size, segment_width):
            if component_size >= (segment_width * self.size_classification_factor):
                return "L"
            elif component_size != 0:
                return "S"
            else:
                return "N"

        feature_string = ""
        for i in range(self.nr_of_divisions):
            segment_comp_sizes = features_for_segments[i]
            segment = segments[i]
            segment_width = segment.shape[0]
            segment_feature_string = ""
            for size in segment_comp_sizes:
                segment_feature_string = (segment_feature_string +
                                          classify_component(size, segment_width))
            feature_string = (feature_string +
                              self.feature_pattern_to_id[segment_feature_string])
        return feature_string

    def extract_feature_strings_for_dir(self,
                                        dir_path,
                                        nr_of_training_examples=10000,
                                        nr_of_test_examples=0):
        image_dir = ImageExampleDir(dir_path)
        images = [image for (label, image) in image_dir]
        nr_of_training_examples = min([nr_of_training_examples, len(images)])
        nr_of_images = len(images)

        test_example_indices = []
        for i in range(nr_of_test_examples):
            random_value_selected = False
            random_number = 0
            while not random_value_selected:
                random_number = int(round(random() * (nr_of_images - 1)))
                if not random_number in test_example_indices:
                    random_value_selected = True
            test_example_indices.append(random_number)

        test_example_indices.sort()
        test_example_indices.reverse()

        feature_strings = [self.extract_feature_string(image) for image in images]
        # take out the test examples
        test_examples = []
        for i in test_example_indices:
            test_examples.append(feature_strings.pop(i))
        if len(feature_strings) > nr_of_training_examples:
            feature_strings = feature_strings[0:nr_of_training_examples]

        return (feature_strings, test_examples)

    def extract_label_examples_tuples_for_library(self, library_path):
        example_dirs = os.listdir(library_path)
        label_example_tuples = []
        for dir_name in example_dirs:
            label = dir_name
            dir = os.path.join(library_path,dir_name)
            examples, test_examples = self.extract_feature_strings_for_dir(dir)
            label_example_tuples.append((label, examples))
        return label_example_tuples

    #
    #
    '''This part is upper contour extract'''

    #
    #
    def extract_upper_contour(self, buffered_image):
        segments = divide_into_segments(self.upper_contour_divisions, buffered_image)

        # Get component sizes for the segments
        features_upper_contour_for_segments = [extract_upper_contour(s)
                                               for s in segments]

        def get_5_feature_upper_contour(fearture):
            if(len(fearture) < 5):
                fearture.append(0)
                get_5_feature_upper_contour(fearture)
            else:
                indexs = range(0, len(fearture), int(math.ceil(float(len(fearture))/5)))
                newFeature = [fearture[index] for index in indexs]
                i = 5 - len(newFeature)
                while i > 0:
                    newFeature.append(fearture[len(fearture) - i])
                    i -= 1
                return newFeature
        for fea in features_upper_contour_for_segments:
            get_5_feature_upper_contour(fea)
        features_upper_contour_for_segments_5getted = [get_5_feature_upper_contour(fea) for fea in features_upper_contour_for_segments]
        def classify_feature(segment, segment_height):
            if segment >= (segment_height * self.upper_contour_threshold):
                return "L"
            else:
                return "S"

        feature_string = ""
        for i in range(self.upper_contour_divisions):
            segment_feature_contour = features_upper_contour_for_segments_5getted[i]
            segment = segments[i]
            segment_height = segment.shape[1]
            segment_feature_string = ""
            for segment in segment_feature_contour:
                segment_feature_string = (segment_feature_string +
                                          classify_feature(segment, segment_height))
            feature_string = (feature_string +
                              self.feature_upper_contour_pattern_to_id[segment_feature_string])
        return feature_string

    def extract_upper_contour_strings_for_dir(self,
                                        dir_path,
                                        nr_of_training_examples=10000,
                                        nr_of_test_examples=0):
        image_dir = ImageExampleDir(dir_path)
        images = [image for (label, image) in image_dir]
        nr_of_training_examples = min([nr_of_training_examples, len(images)])
        nr_of_images = len(images)

        test_example_indices = []
        for i in range(nr_of_test_examples):
            random_value_selected = False
            random_number = 0
            while not random_value_selected:
                random_number = int(round(random() * (nr_of_images - 1)))
                if not random_number in test_example_indices:
                    random_value_selected = True
            test_example_indices.append(random_number)

        test_example_indices.sort()
        test_example_indices.reverse()

        feature_strings = [self.extract_upper_contour(image) for image in images]
        # take out the test examples
        test_examples = []
        for i in test_example_indices:
            test_examples.append(feature_strings.pop(i))
        if len(feature_strings) > nr_of_training_examples:
            feature_strings = feature_strings[0:nr_of_training_examples]

        return (feature_strings, test_examples)

    def extract_label_examples_tuples_for_library_upper_contour(self, library_path):
        example_dirs = os.listdir(library_path)
        label_example_tuples = []
        for dir_name in example_dirs:
            label = dir_name
            dir = os.path.join(library_path,dir_name)
            examples, test_examples = self.extract_upper_contour_strings_for_dir(dir)
            label_example_tuples.append((label, examples))
        return label_example_tuples

    def extract_training_and_test_examples(self,
                                           library_path,
                                           nr_of_training_examples=5000,
                                           nr_of_test_examples=10):
        example_dirs = os.listdir(library_path)
        label_training_example_tuples = []
        label_test_example_tuples = []
        for dir_name in example_dirs:
            label = dir_name
            dir = os.path.join(library_path, dir_name)
            training_examples, test_examples = self.extract_feature_strings_for_dir(dir,
                                                                                    nr_of_training_examples,
                                                                                    nr_of_test_examples)
            label_training_example_tuples.append((label, training_examples))
            label_test_example_tuples.append((label, test_examples))
        return (label_training_example_tuples, label_test_example_tuples)


class TestSimpleImageFeatureExtractor(unittest.TestCase):

    def get_example_image(self):
        example_dir = os.path.join(os.path.abspath('../..'), 'word_examples_for_test', 'A')
        list_image = glob.glob1(example_dir, '*.png')
        image_path_example = os.path.join(example_dir, list_image[0])
        image = cv2.imread(image_path_example, cv2.IMREAD_GRAYSCALE)
        return image

    """def test_extract_feature_string(self):
        image = self.get_example_image()
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=5,
                                                size_classification_factor=4.3)
        feature_string = extractor.extract_feature_string(image)
        print("test_extract_feature_string")
        print(feature_string)

    def test_extract_feature_strings_for_dir(self):
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=7,
                                                size_classification_factor=1.3)
        example_dir_path = os.path.join(os.path.abspath('../..'), 'character_examples', 'I')
        training_examples, test_examples = extractor.extract_feature_strings_for_dir(
            example_dir_path,
            nr_of_training_examples=90,
            nr_of_test_examples=10)
        if len(training_examples) == 90 and len(test_examples) == 10:
            pass
        else:
            raise ValueError("wrong number in returned list")

        print("test_extract_feature_strings_for_dir")
        print(training_examples, test_examples)

    def test_extract_label_examples_tuples_for_library(self):
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=7,
                                                size_classification_factor=1.3)
        library_path = os.path.join(os.path.abspath('../..'), 'character_examples')
        training_examples = extractor.extract_label_examples_tuples_for_library(library_path)
        print("test_extract_label_examples_tuples_for_library")
        print(training_examples)
    """
    """def test_extract_upper_contour(self):
        original_image = self.get_example_image()
        image = scale_to_fill(original_image)
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=5)
        segments = divide_into_segments(5, image)
        for s in segments:
            feature_string = extractor.extract_upper_contour(s)
            component_size_list = extract_sorted_component_size_list(s)
            print(component_size_list)
            print("FeatureString: ",feature_string)
    """

    def test_extract_upper_contour_strings_for_dir(self):
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=7)
        example_dir_path = os.path.join(os.path.abspath('../..'), 'character_examples', 'I')
        training_examples, test_examples = extractor.extract_upper_contour_strings_for_dir(
            example_dir_path,
            nr_of_training_examples=90,
            nr_of_test_examples=10)
        if len(training_examples) == 90 and len(test_examples) == 10:
            pass
        else:
            raise ValueError("wrong number in returned list")

        print("test_extract_orientation_uppper_contour_strings_for_dir")
        print(training_examples, test_examples)

    def test_extract_label_examples_tuples_for_library_upper_contour(self):
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=7)
        library_path = os.path.join(os.path.abspath('../..'), 'character_examples')
        training_examples = extractor.extract_label_examples_tuples_for_library_upper_contour(library_path)
        print("test_extract_label_examples_tuples_for_library_orientation")
        print(training_examples)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_word_']
    unittest.main()