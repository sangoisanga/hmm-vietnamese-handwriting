import glob
import math
import os
import unittest
from random import random

import cv2

from main.feature.feature_manager import FeatureManager
from main.feature.image_example_dir import ImageExampleDir
from main.feature.image_preprocessor import scale_to_fill, divide_into_segments, extract_sorted_component_size_list, \
    extract_orientation_upper_contour, divide_into_segments_new, extract_orientation_lower_contour, \
    extract_upper_contour


class SimpleImageFeatureExtractor(object):
    """
    A class used to extract a sequence of features from an image that
    may be used as training observations for a HMM.
    """
    component_ids = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    component_pattern_to_id = {"LLL": "a",
                               "LLS": "b",
                               "LSS": "c",
                               "LSN": "d",
                               "LLN": "e",
                               "LNN": "f",
                               "SSS": "g",
                               "SSN": "h",
                               "SNN": "i",
                               "NNN": "j"}
    component_pattern = ["LLL", "LLS", "LSS", "LSN", "LLN", "LNN", "SSS", "SSN", "SNN", "NNN"]

    orientation_fm = FeatureManager(2, ['L', 'S', 'N'])
    orientation_pattern_to_id = orientation_fm.get_dictionary()

    upper_contour_fm = FeatureManager(5, ['L', 'S'])
    upper_contour_pattern_to_id = upper_contour_fm.get_dictionary()

    full_fm = FeatureManager(2, ['L', 'S', 'N'], [(5, ['L', 'S'])], component_pattern)
    full_pattern2id = full_fm.get_dictionary(True)

    orientation_extract = "ORIENTATION"
    component_extract = "COMPONENT"
    upper_contour_extract = "UPPER_CONTOUR"
    full_extract = "FULL"

    def __init__(self,
                 nr_of_divisions=7,
                 overlap=None,
                 extract_mode=component_extract,
                 size_classification_factor=1.3,
                 contour_upper_factor=0.5,
                 from_string=None):
        """
        :param contour_upper_factor:
        :param nr_of_divisions: Number of times to divide the image vertically
        :param size_classification_factor: A component in a segment is classified
            as small if the component size is less than "segment_width * size_classification_factor"
            and greater than zero otherwise it is classified as large. Zero size segments are
            classified as none.
        :param overlap: previous segment = overlap * current segment in image segmentation

        """
        if from_string is not None:
            nr_of_divisions1, overlap1, extract_mode1, size_classification_factor1, contour_upper_factor1 = from_string
            self.nr_of_divisions = nr_of_divisions1
            self.overlap = overlap1
            self.extract_mode = extract_mode1
            self.size_classification_factor = size_classification_factor1
            self.contour_upper_factor = contour_upper_factor1
            return

        self.nr_of_divisions = nr_of_divisions
        self.overlap = overlap
        self.extract_mode = extract_mode
        self.size_classification_factor = size_classification_factor
        self.contour_upper_factor = contour_upper_factor

    def get_observer_ids(self):
        if self.extract_mode == self.orientation_extract:
            return [v for k, v in self.orientation_pattern_to_id.items()]
        elif self.extract_mode == self.component_extract:
            return self.component_ids
        elif self.extract_mode == self.upper_contour_extract:
            return [v for k, v in self.upper_contour_pattern_to_id.items()]
        elif self.extract_mode == self.full_extract:
            return [v for k, v in self.full_pattern2id.items()]
        else:
            raise ValueError("Can not detect extract mode")

    def extract_component_string(self, buffered_image):
        """

        :param buffered_image:
        :return:

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

        """

        scaled_image = scale_to_fill(buffered_image)
        segments = []
        if self.overlap is None:
            segments.extend(divide_into_segments(self.nr_of_divisions, scaled_image))
        else:
            segments.extend(divide_into_segments_new(self.nr_of_divisions, scaled_image, self.overlap))
        # Get component sizes for the segments
        features_for_segments = [extract_sorted_component_size_list(s)
                                 for s in segments]

        # Make sure that there are 3 elements on the list for all segmensts
        def make_size_of_list3(lis):
            if len(lis) == 3:
                return lis
            elif len(lis) > 3:
                del lis[len(lis) - 1]
                return make_size_of_list3(lis)
            elif len(lis) < 3:
                lis.append(0)
                return make_size_of_list3(lis)

        features_for_segments = [make_size_of_list3(l)
                                 for l in features_for_segments]

        def classify_component(component_size, seg_width):
            if component_size >= (seg_width * self.size_classification_factor):
                return "L"
            elif component_size != 0:
                return "S"
            else:
                return "N"

        feature_string = []
        raw = []
        for i in range(self.nr_of_divisions):
            segment_comp_sizes = features_for_segments[i]
            segment = segments[i]
            segment_width = segment.shape[0]
            segment_feature_string = ""
            for size in segment_comp_sizes:
                segment_feature_string = (segment_feature_string +
                                          classify_component(size, segment_width))
            raw.append(segment_feature_string)
            feature_string.extend(self.component_pattern_to_id[segment_feature_string])
        return feature_string, raw

    def extract_orientation_upper_contour_string(self, buffered_image):
        scaled_image = scale_to_fill(buffered_image)
        segments = []
        if self.overlap is None:
            segments.extend(divide_into_segments(self.nr_of_divisions, scaled_image))
        else:
            segments.extend(divide_into_segments_new(self.nr_of_divisions, scaled_image, self.overlap))
        # Get component sizes for the segments

        feature = [extract_orientation_upper_contour(s) for s in segments]

        def classify_component(phi):
            if phi > 0:
                return "L"
            elif phi < 0:
                return "S"
            else:
                return "N"

        feature_string = ""

        for i in range(self.nr_of_divisions):
            feature_string = feature_string + classify_component(feature[i])
        return feature_string

    def extract_orientation_lower_contour_string(self, buffered_image):
        scaled_image = scale_to_fill(buffered_image)
        segments = []
        if self.overlap is None:
            segments.extend(divide_into_segments(self.nr_of_divisions, scaled_image))
        else:
            segments.extend(divide_into_segments_new(self.nr_of_divisions, scaled_image, self.overlap))
        # Get component sizes for the segments

        feature = [extract_orientation_lower_contour(s) for s in segments]

        def classify_component(phi):
            if phi > 0:
                return "L"
            elif phi < 0:
                return "S"
            else:
                return "N"

        feature_string = ""

        for i in range(self.nr_of_divisions):
            feature_string = feature_string + classify_component(feature[i])
        return feature_string

    def extract_orientation_string(self, buffered_image):
        """
        :param buffered_image: can xuat ra feature string
        :return: [a-i]+

        feature id | upper contour | lower contour |
        a          |       L       |       L       |
        b          |       L       |       S       |
        c          |       L       |       N       |
        d          |       S       |       L       |
        e          |       S       |       S       |
        f          |       S       |       N       |
        g          |       N       |       L       |
        h          |       N       |       S       |
        i          |       N       |       N       |
        """
        upper_coutour_string = self.extract_orientation_upper_contour_string(buffered_image)
        lower_coutour_string = self.extract_orientation_lower_contour_string(buffered_image)

        orientation_string = []
        raw = []
        for i in range(len(upper_coutour_string)):
            pattern = upper_coutour_string[i] + lower_coutour_string[i]
            raw.append(pattern)
            orientation_string.extend(self.orientation_pattern_to_id[pattern])

        return orientation_string, raw

    def extract_upper_contour_segment(self, buffered_image):
        feature = extract_upper_contour(buffered_image)
        image_height = buffered_image.shape[0]
        number_of_contour = 5

        while (len(feature) < number_of_contour):
            feature.append(0)
        step = int(math.ceil(float(len(feature)) / number_of_contour))

        feature_for_segment = []
        for i in range(0, len(feature), step):
            feature_for_segment.append(feature[i])

        if (len(feature_for_segment) < number_of_contour):
            feature_for_segment.append(feature[len(feature) - 1])

        def classify_component(upper_contour):
            if upper_contour > (image_height * self.contour_upper_factor):
                return "L"
            elif upper_contour <= (image_height * self.contour_upper_factor):
                return "S"

        feature_string = ""

        for i in range(number_of_contour):
            feature_string = feature_string + classify_component(feature[i])
        return feature_string

    def extract_upper_contour_string(self, buffered_image):
        scaled_image = scale_to_fill(buffered_image)
        segments = divide_into_segments(self.nr_of_divisions, scaled_image)

        feature = [self.extract_upper_contour_segment(s) for s in segments]

        feature_string = []
        raw = []
        for i in range(self.nr_of_divisions):
            feature_string = feature_string + [self.upper_contour_pattern_to_id[feature[i]]]
            raw.append(feature[i])
        return feature_string, raw

    def extract_full_feature_string(self, buffered_image):
        scaled_image = scale_to_fill(buffered_image)

        orientation = self.extract_orientation_string(scaled_image)[1]
        upper_contour = self.extract_upper_contour_string(scaled_image)[1]
        component = self.extract_component_string(scaled_image)[1]

        feature_string = []
        for i in range(self.nr_of_divisions):
            feature_string = feature_string + [self.full_pattern2id[orientation[i] + upper_contour[i] + component[i]]]
        return feature_string

    def extract_feature_string(self, buffered_image):

        if self.extract_mode == self.component_extract:
            feature_string = self.extract_component_string(buffered_image)
        elif self.extract_mode == self.orientation_extract:
            feature_string = self.extract_orientation_string(buffered_image)
        elif self.extract_mode == self.upper_contour_extract:
            feature_string = self.extract_upper_contour_string(buffered_image)
        elif self.extract_mode == self.full_extract:
            feature_string = self.extract_full_feature_string(buffered_image)
        else:
            raise ValueError("Not defined extract function!")

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
            dir1 = os.path.join(library_path, dir_name)
            examples, test_examples = self.extract_feature_strings_for_dir(dir1)
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
            dir1 = os.path.join(library_path, dir_name)
            training_examples, test_examples = self.extract_feature_strings_for_dir(dir1,
                                                                                    nr_of_training_examples,
                                                                                    nr_of_test_examples)
            label_training_example_tuples.append((label, training_examples))
            label_test_example_tuples.append((label, test_examples))
        return (label_training_example_tuples, label_test_example_tuples)

    def get_feature_extractor_parameters(self):
        feature_extractor_parameters = (self.nr_of_divisions,
                                        self.overlap,
                                        self.extract_mode,
                                        self.size_classification_factor,
                                        self.contour_upper_factor
                                        )

        return feature_extractor_parameters


class TestSimpleImageFeatureExtractor(unittest.TestCase):

    def get_example_image(self):
        example_dir = os.path.join(os.path.abspath('../..'), 'word_examples_for_test', 'A')
        list_image = glob.glob1(example_dir, '*.png')
        image_path_example = os.path.join(example_dir, list_image[0])
        image = cv2.imread(image_path_example, cv2.IMREAD_GRAYSCALE)
        return image

    def test_extract_component_string(self):
        image = self.get_example_image()
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=5,
                                                size_classification_factor=4.3,
                                                overlap=0.5)
        feature_string = extractor.extract_component_string(image)
        print("test_extract_component_string")
        print(feature_string)

    def test_extract_component_strings_for_dir(self):
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=7,
                                                size_classification_factor=1.3,
                                                overlap=0.5,
                                                extract_mode=SimpleImageFeatureExtractor.upper_contour_extract)
        example_dir_path = os.path.join(os.path.abspath('../..'), 'character_examples', 'A')
        training_examples, test_examples = extractor.extract_feature_strings_for_dir(
            example_dir_path,
            nr_of_training_examples=90,
            nr_of_test_examples=10)
        if len(training_examples) == 90 and len(test_examples) == 10:
            pass
        else:
            raise ValueError("wrong number in returned list")

        print("test_extract_component_strings_for_dir")
        print(training_examples, test_examples)

    def test_extract_label_examples_tuples_for_library(self):
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=7, overlap=0.5)
        library_path = os.path.join(os.path.abspath('../..'), 'character_examples')
        training_examples = extractor.extract_label_examples_tuples_for_library(library_path)
        print("test_extract_label_examples_tuples_for_library")
        print(training_examples)

    def test_extract_orientation_upper_contour_string(self):
        image = self.get_example_image()
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=5, overlap=0.5)
        feature_string = extractor.extract_orientation_upper_contour_string(image)
        print("test_extract_orientation_upper_contour_string")
        print(feature_string)

    def test_extract_orientation_strings_for_dir(self):
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=7, overlap=0.5,
                                                extract_mode=SimpleImageFeatureExtractor.orientation_extract)
        example_dir_path = os.path.join(os.path.abspath('../..'), 'character_examples', 'I')
        training_examples, test_examples = extractor.extract_feature_strings_for_dir(
            example_dir_path,
            nr_of_training_examples=90,
            nr_of_test_examples=10)
        if len(training_examples) == 90 and len(test_examples) == 10:
            pass
        else:
            raise ValueError("wrong number in returned list")

        print("test_extract_orientation_strings_for_dir")
        print(training_examples, test_examples)

    def test_extract_label_examples_tuples_for_library_orientation(self):
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=7, overlap=0.5,
                                                extract_mode=SimpleImageFeatureExtractor.orientation_extract)
        library_path = os.path.join(os.path.abspath('../..'), 'character_examples')
        training_examples = extractor.extract_label_examples_tuples_for_library(library_path)
        print("test_extract_label_examples_tuples_for_library_orientation")
        print(training_examples)

    def test_extract_orientation_lower_contour_string(self):
        image = self.get_example_image()
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=5, overlap=0.5)
        feature_string = extractor.extract_orientation_lower_contour_string(image)
        print("test_extract_orientation_lower_contour_string")
        print(feature_string)

    def test_extract_orientation_string(self):
        image = self.get_example_image()
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=5, overlap=0.5)
        feature_string = extractor.extract_orientation_string(image)
        print("test_extract_orientation_string")
        print(feature_string)

    def test_extract_upper_contour_string(self):
        image = self.get_example_image()
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=5, overlap=0.5)
        feature_string = extractor.extract_upper_contour_string(image)
        print("test_extract_upper_contour_string")
        print(feature_string)

    def test_upper_contour_pattern_to_id(self):
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=5, overlap=0.5)
        print extractor.upper_contour_pattern_to_id
        print len(extractor.upper_contour_pattern_to_id)

    def test_extract_full_feature_string(self):
        image = self.get_example_image()
        extractor = SimpleImageFeatureExtractor(nr_of_divisions=5, overlap=0.5,
                                                extract_mode=SimpleImageFeatureExtractor.full_extract)
        feature_string = extractor.extract_full_feature_string(image)
        print("test_extract_full_feature_string")
        print(feature_string)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_word_']
    unittest.main()
