import os
import natsort
import mnist
import cv2
import numpy as np
import glob


######### Corner finding method ################
def c_rect(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    Big_1_3 = c.item(0) - c.item(1)
    Big_2_4 = c.item(0) + c.item(1)
    Little_1_3 = c.item(0) - c.item(1)
    Little_2_4 = c.item(0) + c.item(1)
    c_1 = None
    c_2 = None
    c_3 = None
    c_4 = None

    for point in c:
        if point.item(0) + point.item(1) >= Big_2_4:
            c_4 = point
            Big_2_4 = point.item(0) + point.item(1)
        if point.item(0) + point.item(1) <= Little_2_4:
            c_2 = point
            Little_2_4 = point.item(0) + point.item(1)
        if point.item(0) - point.item(1) >= Big_1_3:
            c_3 = point
            Big_1_3 = point.item(0) - point.item(1)
        if point.item(0) - point.item(1) <= Little_1_3:
            c_1 = point
            Little_1_3 = point.item(0) - point.item(1)
    # print(c_1, " ", c_2, " ", c_3, " ", c_4)
    return c_1, c_2, c_3, c_4


def detector(image):
    kernel_size = np.ones((3, 3), np.uint8)
    img_copy = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.fastNlMeansDenoising(image, None, 10, 10, 5)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C, 13, 2)
    image = cv2.bitwise_not(image)
    image = cv2.dilate(image, kernel_size, iterations=1)
    image = cv2.erode(image, kernel_size, iterations=1)
    ################### Out side of the box ######################
    c_1, c_2, c_3, c_4 = c_rect(image)
    # cv2.line(img_copy, (c_1.item(0), c_1.item(1)), (c_2.item(0), c_2.item(1)), (255, 255, 0), 3, cv2.LINE_AA)
    # cv2.line(img_copy, (c_2.item(0), c_2.item(1)), (c_3.item(0), c_3.item(1)), (255, 255, 0), 3, cv2.LINE_AA)
    # cv2.line(img_copy, (c_3.item(0), c_3.item(1)), (c_4.item(0), c_4.item(1)), (255, 255, 0), 3, cv2.LINE_AA)
    # cv2.line(img_copy, (c_4.item(0), c_4.item(1)), (c_1.item(0), c_1.item(1)), (255, 255, 0), 3, cv2.LINE_AA)
    ################## Inside of the box #######################
    # x_4_1 = int((c_4.item(0) - c_1.item(0)) / 9)
    # y_4_1 = int((c_4.item(1) - c_1.item(1)) / 9)
    # x_3_2 = int((c_3.item(0) - c_2.item(0)) / 9)
    # y_3_2 = int((c_3.item(1) - c_2.item(1)) / 9)
    # x_2_1 = int((c_2.item(0) - c_1.item(0)) / 9)
    # y_2_1 = int((c_1.item(1) - c_2.item(1)) / 9)
    # x_4_3 = int((c_4.item(0) - c_3.item(0)) / 9)
    # y_4_3 = int((c_4.item(1) - c_3.item(1)) / 9)

    # for i in range(1, 9):
    #     cv2.line(img_copy, (c_1.item(0) + (x_4_1 * i), c_1.item(1) + (y_4_1 * i)),
    #              (c_2.item(0) + (x_3_2 * i), c_2.item(1) + (y_3_2 * i)), (255, 255, 0), 3, cv2.LINE_AA)
    #     cv2.line(img_copy, (c_1.item(0) + (x_2_1 * i), c_2.item(1) + (y_2_1 * i)),
    #              (c_3.item(0) + (x_4_3 * i), c_3.item(1) + (y_4_3 * i)), (255, 255, 0), 3, cv2.LINE_AA)
    ########## CROPING PART ########################
    width, height = 480, 480
    pts1 = np.float32([[c_2.item(0), c_2.item(1)], [c_3.item(0), c_3.item(1)], [c_1.item(0), c_1.item(1)],
                       [c_4.item(0), c_4.item(1)]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img_copy, matrix, (width, height))

    img_list = list()
    y = 0
    for i in range(9):
        x = 0
        for z in range(9):
            box = imgOutput[y:y + 53, x:x + 53]
            box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
            box = cv2.fastNlMeansDenoising(box, None, 10, 7, 5)
            box = cv2.adaptiveThreshold(box, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C, 7, 2)
            box = cv2.bitwise_not(box)
            crop_img = box[10:46, 10:46]
            # cv2.imshow("dada", crop_img)
            # cv2.waitKey()
            res_box = cv2.resize(crop_img, (28, 28))
            img_list.append(res_box)
            x += 53
        y += 53

    return imgOutput, img_list


images = mnist.train_images().reshape(60000, 784)
labels = mnist.train_labels().reshape(60000, 1).astype(np.float32)
images_test = mnist.test_images().reshape(10000, 784)
labels_test = mnist.test_labels().reshape(10000, 1).astype(np.float32)



# PCA aplicaion on data set
def PCA(img, img_test, k):
    # prep_data
    mean = np.mean(img, axis=0)
    img_test = img_test - mean
    img = img - mean
    # process_data
    cov = np.cov(img.T)
    eig_values, eig_vectors = np.linalg.eig(cov)
    eig_vectors = eig_vectors.T
    index = np.argsort(eig_values)[::-1]
    eig_values = eig_values[index]
    eig_vectors = eig_vectors[index]
    comp = eig_vectors[0:k]
    # forward_data
    out = np.dot(img, comp.T)
    out_test = np.dot(img_test, comp.T)
    return out, out_test



# Converting PCA aplied dataset complex128 to float32
data_train, data_test = PCA(images, images_test, 25)
data_train, data_test = data_train.real, data_test.real
data_train, data_test = data_train.astype(np.float32), data_test.astype(np.float32)


def SudokuDigitDetector(img):
    img, img_l = detector(img)
    out = [10 for _ in range(81)]
    test = np.asarray(img_l)
    test = test.reshape(81, 784)
    _, data_sudo_test = PCA(images, test, 25)
    data_sudo_test = data_sudo_test.real
    data_sudo_test = data_sudo_test.astype(np.float32)
    ret, result, neighbours, dist = model.findNearest(data_sudo_test, k=5)
    for index in range(81):
        if np.count_nonzero(img_l[index] == 0) > 700:
            out.pop(index)
            out.insert(index, 0)
        else:
            out.pop(index)
            out.insert(index, result[index])
        index += 1
    out = np.asarray(out)
    out = out.reshape(9, 9).astype(int)
    print(out)
    return out


def sudokuAcc(gt, out):
    return (gt == out).sum() / gt.size * 100


if __name__ == "__main__":

    # MNIST experiments:
    model = cv2.ml.KNearest_create()
    model.train(data_train, cv2.ml.ROW_SAMPLE, labels)
    ret, result, neighbours, dist = model.findNearest(data_test, k=5)

    matches = result == labels_test
    correct = np.count_nonzero(matches)
    accuracy = correct * 100.0 / result.size
    print("-PCA aplied-Mnist KNN resault: ", accuracy)

    # Sudoku Experiments:
    image_dirs = 'C:/Users/Muco/Desktop/images/*.jpg'
    data_dirs = 'C:/Users/Muco/Desktop/images/*.dat'
    IMAGE_DIRS = natsort.natsorted(glob.glob(image_dirs))
    DATA_DIRS = natsort.natsorted(glob.glob(data_dirs))
    total_acc = 0
    # Loop over all images and ground truth
    for i, (img_dir, data_dir) in enumerate(zip(IMAGE_DIRS, DATA_DIRS)):
        # Define your variables etc.:
        image_name = os.path.basename(img_dir)
        gt = np.genfromtxt(data_dir, skip_header=2, dtype=int, delimiter=' ')
        img = cv2.imread(img_dir)
        # implement this function, inputs img, outputs in the same format as data 9x9 numpy array.
        output = SudokuDigitDetector(img)
        print(sudokuAcc(gt, output))
        total_acc = total_acc + sudokuAcc(gt, output)
        gt = gt.reshape(81, 1)
        output = output.reshape(81, 1)

    print("Sudoku dataset accuracy: {}".format(total_acc / (i + 1)))
