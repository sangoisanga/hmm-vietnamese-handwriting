import glob
import os
from os.path import join

from main.character.character_classifier import CharacterClassifier

base_dir = os.path.abspath('../')
character_data_path = join(base_dir, 'model_creation', 'character_classifier.dat')
character_classifier_file = open(character_data_path, 'r')
character_classifier = CharacterClassifier(from_string_string=character_classifier_file.read())
character_classifier_file.close()

test_dir = join(base_dir, 'test_of_tuan')

list_test = glob.glob1(test_dir, '*.png')

print character_data_path
