import glob
import os
import time
import unittest

import cv2

current_milli_time = lambda: int(round(time.time() * 1000))

class ImageExampleDir:
    '''
    Handler for a directory containing image examples
    '''

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.reload_elements()

    def reload_elements(self):
        files_in_dir = os.listdir(self.dir_path)
        self.example_list = [elem for elem in files_in_dir if elem.endswith(".png") and ("_" in elem)]

    def save_example(self, label, image_np_array):
        output_file_name = label + "_" + str(current_milli_time) + ".png"
        save_path = os.path.join(self.dir_path,output_file_name)
        for image in image_np_array:
            cv2.imwrite(save_path,image_np_array)

    def __iter__(self):
        dir_path = self.dir_path

        class ImageExampleDirIter:

            def next(self):
                image_file_name = self.iterator.next()
                # Find example label
                label = image_file_name[0:image_file_name.index('_')]
                # Get the image as an image buffer
                image_path = os.path.join(dir_path,image_file_name)
                buffered_image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                # Return label and image tuple
                return (label, buffered_image)

            def __init__(self, iter_list):
                self.iterator = iter_list.__iter__()

        self.reload_elements()
        return ImageExampleDirIter(self.example_list)


class TestImageExampleDir(unittest.TestCase):

    def test_image_example_dir_iteration(self):
        example_dir = os.path.join(os.path.abspath('../..'), 'word_examples_for_test', 'A')
        list_image = glob.glob1(example_dir, '*.png')
        image_example_dir = ImageExampleDir(example_dir.getCanonicalPath())
        for label, image in image_example_dir:
            if label != "A":
                raise ValueError("The label of the examples in this dir should be A")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_word_']
    unittest.main()
