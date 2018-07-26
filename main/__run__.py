import glob
import os
from os.path import join

import cv2

from main.character.character_classifier import CharacterClassifier

base_dir = os.path.abspath('../')
character_data_path = join(base_dir, 'main', 'model_creation', 'character_classifier.dat')
character_classifier_file = open(character_data_path, 'r')
character_classifier = CharacterClassifier(from_string_string=character_classifier_file.read())
character_classifier_file.close()

test_dir = join(base_dir, 'test_of_tuan')

list_test = glob.glob1(test_dir, '*.png')

for img in list_test:
    image = cv2.imread(join(test_dir, img), cv2.IMREAD_GRAYSCALE)
    char = character_classifier.classify_image(image)
    print "Picture " + img + " is: " + char[0][0] + ", " + char[1][0] + ", " + char[2][0] + "\n"
