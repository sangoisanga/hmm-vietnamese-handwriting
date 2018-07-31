import glob
import os
from os.path import join

import cv2

from main.character.character_classifier import CharacterClassifier

base_dir = os.path.abspath('../')

'''
character_data_path_4_0 = join(base_dir, 'main', 'model_creation', 'character_classifier_4-0.dat')
character_classifier_file_4_0 = open(character_data_path_4_0, 'r')
character_classifier_4_0 = CharacterClassifier(from_string_string=character_classifier_file_4_0.read())
character_classifier_file_4_0.close()

character_data_path_4_2 = join(base_dir, 'main', 'model_creation', 'character_classifier_4-2.dat')
character_classifier_file_4_2 = open(character_data_path_4_2, 'r')
character_classifier_4_2 = CharacterClassifier(from_string_string=character_classifier_file_4_2.read())
character_classifier_file_4_2.close()

character_data_path_4_4 = join(base_dir, 'main', 'model_creation', 'character_classifier_4-4.dat')
character_classifier_file_4_4 = open(character_data_path_4_4, 'r')
character_classifier_4_4 = CharacterClassifier(from_string_string=character_classifier_file_4_4.read())
character_classifier_file_4_4.close()

character_data_path_4_6 = join(base_dir, 'main', 'model_creation', 'character_classifier_4-6.dat')
character_classifier_file_4_6 = open(character_data_path_4_6, 'r')
character_classifier_4_6 = CharacterClassifier(from_string_string=character_classifier_file_4_6.read())
character_classifier_file_4_6.close()
'''
character_data_path_3_6 = join(base_dir, 'main', 'model_creation', 'character_classifier_11_segments_4_6_cf.dat')
character_classifier_file_3_6 = open(character_data_path_3_6, 'r')
character_classifier_3_6 = CharacterClassifier(from_string_string=character_classifier_file_3_6.read())
character_classifier_file_3_6.close()



test_dir = join(base_dir, 'test_of_tuan')

list_test = glob.glob1(test_dir, '*.png')

for img in list_test:

    image = cv2.imread(join(test_dir, img), cv2.IMREAD_GRAYSCALE)

    print  "Picture " + img + "\n"
    char = character_classifier_3_6.classify_image(image)

    # print char[0][1] == char[1][1]

    print "Factor 3.6" + " is: " \
          + char[0][0][0] + ': ' + str(char[0][1]) + ", " \
          + char[1][0][0] + ': ' + str(char[1][1]) + ", " \
          + char[2][0][0] + ': ' + str(char[0][1]) + "\n"

    '''
    char = character_classifier_4_6.classify_image(image)
    print "Factor 4.6" + " is: " \
          + char[0][0][0] + ': ' + str(char[0][1]) + ", " \
          + char[1][0][0] + ': ' + str(char[1][1]) + ", " \
          + char[2][0][0] + ': ' + str(char[0][1]) + "\n"
    '''
