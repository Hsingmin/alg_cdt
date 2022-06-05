import cv2 as cv
import math
import matplotlib.pyplot as plt

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

def get_inscribed_circle_coord(width, height, x):
    r = width // 2 if width < height else height // 2
    # (x - r) * (x - r) + (y - r) * (y - r) = r * r
    y1 = r - math.sqrt(r * r - ((x - r) * (x - r)))
    y2 = r + math.sqrt(r * r - ((x - r) * (x - r)))
    return (y1, y2)

def sample_and_diff(img, sample_rate):
    height = img.shape[0]
    width = img.shape[1]
    power2_delta = 0
    for x in range(width - 1):
        (bottom, top) = get_drawing_point_coord(img, x)
        (y1, y2) = get_inscribed_circle_coord(width, height, x)
        power2_delta += (y1 - bottom) * (y1 - bottom) + (y2 - top) * (y2 - top)
    r = width // 2 if width < height else height // 2    
    power2_delta = power2_delta / (sample_rate * r * r)
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
    score_circle = sample_and_diff(img, img.shape[1])
    print("get score of circle {}: {}".format(img_dir, score_circle))
    # cv.imwrite("../tmp/" + img_dir.split("/")[2].split(".")[0] + "_bin.png", img)

if __name__ == "__main__":
    refer_dir = "../imgs/early.png"
    test_dir = "../imgs/late.png"
    print("================= coordinate the standard cdt image ===============")
    coordinate_img(refer_dir)
    print("================= coordinate the test cdt image ===============")
    coordinate_img(test_dir)
