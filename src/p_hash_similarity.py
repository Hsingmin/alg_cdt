import cv


def compare_img_p_hash(img1, img2):
    """
    Get the similarity of two pictures via pHash
    Generally, when:
    ham_dist == 0 -> particularly like
    ham_dist < 5  -> very like
    ham_dist > 10 -> different image​
    Attention: this is not accurate compare_img_hist() method, so use hist() method to auxiliary comparision.
    This method is always used for graphical search applications, such as Google Image(Use photo to search photo)
    :param img1:
    :param img2:
    :return:
    """
    hash_img1 = get_img_p_hash(img1)
    hash_img2 = get_img_p_hash(img2)
    ​
    return ham_dist(hash_img1, hash_img2)
    ​
    ​


def get_img_p_hash(img):
    """
        Get the pHash value of the image, pHash : Perceptual hash algorithm(感知哈希算法)                                  ​
    :param img: img in MAT format(img = cv2.imread(image))
    :return: pHash value
    """
    hash_len = 32
    ​
    # GET Gray image
    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ​
    # Resize image, use the different way to get the best result
    resize_gray_img = cv.resize(
        gray_img, (hash_len, hash_len), cv.INTER_AREA)
    # resize_gray_img = cv.resize(gray_img, (hash_len, hash_len), cv.INTER_LANCZOS4)
    # resize_gray_img = cv.resize(gray_img, (hash_len, hash_len), cv.INTER_LINEAR)
    # resize_gray_img = cv.resize(gray_img, (hash_len, hash_len), cv.INTER_NEAREST)
    # resize_gray_img = cv.resize(gray_img, (hash_len, hash_len), cv.INTER_CUBIC)
    ​
    # Change the int of image to float, for better DCT
    h, w = resize_gray_img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = resize_gray_img
    ​
    # DCT: Discrete cosine transform(离散余弦变换)
    vis1 = cv.dct(cv.dct(vis0))
    vis1.resize(hash_len, hash_len)
    img_list = vis1.flatten()
    ​
    # Calculate the avg value
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = []
    for i in img_list:
        if i < avg:
            tmp = '0'
            else:
                tmp = '1'
                avg_list.append(tmp)
                ​
                # Calculate the hash value
                p_hash_str = ''
                for x in range(0, hash_len * hash_len, 4):
                    p_hash_str += '%x' % int(
                        ''.join(avg_list[x:x + 4]), 2)
                    return p_hash_str
                    ​
                    ​


def ham_dist(x, y):
    """
        Get the hamming distance of two values.
    hamming distance(汉明距)
    :param x:
    :param y:
    :return: the hamming distance
    """
    assert len(x) == len(y)
    return sum([ch1 != ch2 for ch1, ch2 in zip(x, y)])
