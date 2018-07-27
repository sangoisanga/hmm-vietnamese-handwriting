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

    pic1 = "H.jpg"
    name = pic1[0:pic1.index('.')]
    # name = "V"
    img = cv2.imread(pic1)

    '''
    pic2 = "V2.jpg"
    img2 = cv2.imread(pic2)

    pic3 = "V3.jpg"
    img3 = cv2.imread(pic3)


    a1 = devide_width(img, 2)
    a2 = devide_width(img2, 5)
    a3 = devide_width(img3, 3)

    a = a1 +a2 +a3
    '''

    a = devide_width(img, 11)
    temp = []
    for i in range(len(a)):
        temp.extend(devide_hight(a[i], 4))


    save_image(name, temp)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
