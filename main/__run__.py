import glob
import os
from os.path import join

import cv2

from main.character.character_classifier import CharacterClassifier


def loadCharaterClassifier(name):
    base_dir = os.path.abspath('../')
    character_data_path = join(base_dir, 'main', 'model_creation', name)
    character_classifier_file = open(character_data_path, 'r')
    character_classifier = CharacterClassifier(from_string_string=character_classifier_file.read())
    character_classifier_file.close()

    return character_classifier


if __name__ == "__main__":

    test_dir = join(os.path.abspath('../'), 'test_of_tuan')
    list_test = glob.glob1(test_dir, '*.png')

    #character = loadCharaterClassifier('character_classifier_orientation.dat')
    character = loadCharaterClassifier('character_classifier_orientation_full.dat')

    for img in list_test:
        image = cv2.imread(join(test_dir, img), cv2.IMREAD_GRAYSCALE)

        print  "Picture " + img + "\n"
        # print char[0][1] == char[1][1]

        char = character.classify_image(image)
        '''
        print "Orientation" + " is: " \
              + char[0][0][0] + ': ' + str(char[0][1]) + ", " \
              + char[1][0][0] + ': ' + str(char[1][1]) + ", " \
              + char[2][0][0] + ': ' + str(char[0][1]) + "\n"
        '''
        print "Orientation" + " is: " \
              + char[0][0][0] + ", " \
              + char[1][0][0] + ", " \
              + char[2][0][0] + "\n"
