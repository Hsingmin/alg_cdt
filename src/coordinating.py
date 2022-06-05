import cv2 as cv
import math
import matplotlib.pyplot as plt

def get_drawing_range(img, axis):
    height = img.shape[0]
    width = img.shape[1]
    right = 0
    left = width - 1
    bottom = height - 1
    top = 0
    for row in range(height - 1):
        for col in range(width - 1):
            if img[row][col] != 0:
                continue
            if row < (height / 2):
                if row < bottom:
                    bottom = row
            else:
                if row > top:
                    top = row
            if col < (width / 2):
                if col < left:
                    left = col
            else:
                if col > right:
                    right = col
    if axis == 0:
        return (left, right)
    if axis == 1:
        return (bottom, top)
    return (0, 0)

def get_circle_radius(img):
    (left, right) = get_drawing_range(img, 0)
    (bottom, top) = get_drawing_range(img, 1)
    radius = (top - bottom) / 2 if ((top - bottom) / 2) > ((right - left) / 2) else (right - left) / 2
    return (radius, (right - left) / 2, (top - bottom) / 2)

def get_drawing_point_coord(img, x):
    height = img.shape[0]
    width = img.shape[1]
    bottom = height - 1
    top = 0
    for y in range(height - 1):
        if img[y][x] != 0:
            continue
        if y < (height / 2):
            # update bottom
            if y < bottom:
                bottom = y
        else:
            if y > top:
                top = y
    return (bottom, top)

def get_inscribed_circle_coord(origin_x, origin_y, r, x):
    # (x - r) * (x - r) + (y - r) * (y - r) = r * r
    # print("origin_x = {}, origin_y = {}, r = {}, x = {}".format(origin_x, origin_y, r, x))
    if r * r < ((x - origin_x) * (x - origin_x)):
        return (origin_y, origin_y)
    y1 = origin_y - math.sqrt(r * r - ((x - origin_x) * (x - origin_x)))
    y2 = origin_y + math.sqrt(r * r - ((x - origin_x) * (x - origin_x)))
    return (y1, y2)

def sample_and_diff(img):
    height = img.shape[0]
    width = img.shape[1]
    power2_delta = 0
    (radius, origin_x, origin_y) = get_circle_radius(img)
    (left, right) = get_drawing_range(img, 0)
    for x in range(left, right):
        (bottom, top) = get_drawing_point_coord(img, x)
        (y1, y2) = get_inscribed_circle_coord(origin_x, origin_y, radius, x)
        power2_delta += (y1 - bottom) * (y1 - bottom) + (y2 - top) * (y2 - top)
    power2_delta = power2_delta / ((right - left) * radius * radius)
    return power2_delta

def coordinate_img(img_dir):
    """
    coordinate image
    """
    img = cv.imread(img_dir)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    img = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    # Get the histogram data of image 1, then using normalize the picture for better compare
    print("=========== img ===========")
    # print(img)
    height = img.shape[0]
    width = img.shape[1]
    print("img with size {} x {}".format(width, height))
    left = 0
    right = 0
    top = 0
    bottom = 0
    eer = 0.02 # edge erase ratio
    for row in range(height - 1):
        for col in range(width - 1):
            if (col < width * eer) or (col > width * (1 - eer)) or (row > height * (1 - eer)) or (row < height * eer):
                img[row][col] = 255
                # print("img[{}][{}] = {}".format(col, row, img[col][row]))
    score_circle = sample_and_diff(img)
    print("get score of circle {}: {}".format(img_dir, score_circle))
    # cv.imwrite("../tmp/" + img_dir.split("/")[2].split(".")[0] + "_bin.png", img)

if __name__ == "__main__":
    refer_dir = "../imgs/1.png"
    test_dir = "../imgs/late.png"
    print("================= coordinate the standard cdt image ===============")
    coordinate_img(refer_dir)
    print("================= coordinate the test cdt image ===============")
    coordinate_img(test_dir)
