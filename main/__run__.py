import glob
import os
from os.path import join

import cv2

from main.character.character_classifier import CharacterClassifier


def loadCharaterClassifier(name):
    """
    :param name: Path of data file
    :return: Return CharacterClassifier object, created from data.
    """
    base_dir = os.path.abspath('../')
    character_data_path = join(base_dir, 'main', 'model_creation', name)
    character_classifier_file = open(character_data_path, 'r')
    character_classifier = CharacterClassifier(
        from_string_string=character_classifier_file.read())
    character_classifier_file.close()

    return character_classifier


if __name__ == "__main__":

    # Path of test data, read all file image *.png
    test_dir = join(os.path.abspath('../'), 'test_image')
    list_test = glob.glob1(test_dir, '*.png')

    # character = loadCharaterClassifier('character_classifier_new_data.dat')
    # character = loadCharaterClassifier('character_classifier_orientation_new.dat')
    character = loadCharaterClassifier('character_classifier_full.dat')

    list_test.sort()
    for img in list_test:
        image = cv2.imread(join(test_dir, img), cv2.IMREAD_GRAYSCALE)
        print "Picture " + img + ""
        char = character.classify_image(image)
        print "Top 3 results are: " \
              + char[0][0][0] + ", " \
              + char[1][0][0] + ", " \
              + char[2][0][0] + "\n"
