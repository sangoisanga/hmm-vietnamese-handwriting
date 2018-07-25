import glob
import os
import sys
import unittest
import cv2


def scale_to_fill(buffered_image):  # np array 1 channel, gray scale
    height, width = buffered_image.shape[:2]

    # Get extreem values from the image
    max_x = 0
    min_x = width
    max_y = 0
    min_y = height
    for x in range(0, width):
        for y in range(0, height):
            color = buffered_image[x][y] == 0
            if color:
                if x > max_x:
                    max_x = x
                if x < min_x:
                    min_x = x
                if y > max_y:
                    max_y = y
                if y < min_y:
                    min_y = y
    # Cut out the part of image containing colored pixels
    sub_image = buffered_image[min_x:max_x, min_y:max_y]

    # Scale the image
    resize_image = cv2.resize(sub_image, (width, height), interpolation=cv2.INTER_CUBIC)
    return resize_image


def divide_into_segments(nr_of_segments, image_buffer):
    height, width = image_buffer.shape
    segment_width = width / nr_of_segments

    def create_segment(start_pos):
        end = start_pos + segment_width
        if end > width:
            this_segment_with = segment_width - (end - width)
        elif (width - end - segment_width) < 0:
            this_segment_with = width - start_pos
        else:
            this_segment_with = segment_width
        seg = image_buffer[0:height, start_pos:start_pos + this_segment_with]
        return seg

    segment_starts = range(0, width, segment_width)
    if len(segment_starts) > nr_of_segments:
        del segment_starts[len(segment_starts) - 1]
    segments = [create_segment(s) for s in segment_starts]
    return segments


def extract_sorted_component_size_list(image_buffer):
    # Search for unprocessed colored pixels and find the component
    height, width = image_buffer.shape[:2]

    # make sure we don't run out of stack space
    old_rec_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(width * height)
    # Remember which pixels have been processed
    processed_colored_pixels = set()

    def neighbour_pixels(pixel):
        x, y = pixel
        neighbours = [(x - 1, y - 1),
                      (x - 1, y),
                      (x - 1, y + 1),
                      (x, y - 1),
                      (x, y + 1),
                      (x + 1, y - 1),
                      (x + 1, y),
                      (x + 1, y + 1)]
        valid_neighbours = [(x, y) for (x, y) in neighbours
                            if (0 <= x < height and
                                0 <= y < width)]
        return valid_neighbours

    def find_component_length(start_pixel):
        x, y = start_pixel
        if image_buffer[x][y] == 255:
            return 0
        elif start_pixel in processed_colored_pixels:
            return 0
        else:
            processed_colored_pixels.add(start_pixel)
            neighbours = neighbour_pixels(start_pixel)

            lengths_of_neighbour_components = [find_component_length(p)
                                               for p in neighbours]
            return 1 + sum(lengths_of_neighbour_components)

    component_lengths = [length for length in
                         [find_component_length((x, y)) for x in range(height) for y in range(width)]
                         if (length > 0)]
    # Set stack limit back to normal
    sys.setrecursionlimit(old_rec_limit)
    # Component lengths shall be sorted with the largest first
    component_lengths.sort()
    component_lengths.reverse()
    return component_lengths


class TestImagePreprocessor(unittest.TestCase):

    def get_example_image(self):
        example_dir = os.path.join(os.path.abspath('../..'), 'character_examples', 'A')
        list_image = glob.glob1(example_dir, '*.png')
        image_path_example = os.path.join(example_dir, list_image[0])
        image = cv2.imread(image_path_example, cv2.IMREAD_GRAYSCALE)
        return image

    def write_image_to_disk(self, image_name, image):
        example_dir = os.path.join(os.path.abspath('../..'), 'test_data')
        image_path = os.path.join(example_dir, image_name)
        cv2.imwrite(image_path, image)

    def test_image_scale_image(self):
        image = self.get_example_image()
        scaled_image = scale_to_fill(image)
        # Print image to disk to test how it looks like
        self.write_image_to_disk("test.png", scaled_image)

    def test_divide_into_segments(self):
        orginal_image = self.get_example_image()
        image = scale_to_fill(orginal_image)
        segments = divide_into_segments(5, image)
        i = 0
        for s in segments:
            self.write_image_to_disk("segment" + str(i) + ".png", s)
            i = i + 1

    def test_extract_sorted_component_size_list(self):
        orginal_image = self.get_example_image()
        image = scale_to_fill(orginal_image)
        segments = divide_into_segments(5, image)
        for s in segments:
            component_size_list = extract_sorted_component_size_list(s)
            print(component_size_list)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_word_']
    unittest.main()
