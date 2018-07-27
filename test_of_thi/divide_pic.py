import cv2


def devide_hight(img, num_height):
    height, width = img.shape[:2]
    segment_height = height / num_height

    temp = []

    for i in range(0, height, segment_height):
        if (i + segment_height) <= height:
            temp.append(img[i:i + segment_height, 0:width])

    return temp


def devide_width(img, num_width):
    height, width = img.shape[:2]

    segment_width = width / num_width

    temp = []

    for i in range(0, width, segment_width):
        if (i + segment_width) <= width:
            temp.append(img[0:height, i:i + segment_width])

    return temp


def save_image(name, list_image):
    i = 0
    for img in list_image:
        if i < 10:
            label = str(0) + str(i)
        else:
            label = str(i)
        file_name = name + label + ".png"
        cv2.imwrite(file_name, img)
        i += 1


if __name__ == "__main__":

    pic = "M.jpg"
    name = pic[0:pic.index('.')]
    img = cv2.imread(pic)

    a = devide_width(img, 10)

    temp = []
    for i in range(10):
        temp.extend(devide_hight(a[i], 4))

    save_image(name, temp)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
