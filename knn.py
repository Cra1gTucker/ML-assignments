# -*- coding: utf-8 -*-
import numpy as np
from converters import convert_from_file

def main():
    training_img = convert_from_file('train-images.idx3-ubyte')
    training_lab = convert_from_file('train-labels.idx1-ubyte')
    test_img = convert_from_file('t10k-images.idx3-ubyte')
    test_lab = convert_from_file('t10k-labels.idx1-ubyte')
    predict_lab = []
    k = 7
    error_counter = 0
    for idx, img in enumerate(test_img):
        knn = []
        for i in range(0, k):
            knn.append((i, np.linalg.norm(img - training_img[i]), training_lab[i]))
        for i in range(k, len(training_img)):
            if np.linalg.norm(img - training_img[i]) < knn[k - 1][1]:
                knn.insert(0, (i, np.linalg.norm(img - training_img[i]), training_lab[i]))
                knn.pop()
                knn.sort(key= lambda x:x[1])
        bucket = [0,0,0,0,0,0,0,0,0,0]
        for neighbor in knn:
            bucket[neighbor[2]] += 1
        predict_lab.append(np.argmax(bucket))
        if predict_lab[-1] != test_lab[idx]:
            error_counter += 1
            print("prediction wrong for image No." + str(idx))
        else:
            print("prediction correct for image No." + str(idx))
    print(str(error_counter) + " out of " + str(len(predict_lab)) + " predictions are wrong.")

if __name__ == '__main__':
    main()
