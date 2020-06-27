from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import numpy as np
import time
from converters import convert_from_file


trainData = convert_from_file('train-images.idx3-ubyte').reshape([60000, 784])
trainLabels = convert_from_file('train-labels.idx1-ubyte')
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size = 0.1, random_state = 42)

testData = convert_from_file('t10k-images.idx3-ubyte').reshape([10000, 784])
testLabels = convert_from_file('t10k-labels.idx1-ubyte')
# Checking sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))


# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 8, 2)
accuracies = []
timings = []
bestScore = 0
# loop over kVals
for k in kVals:
    # train the classifier with the current value of `k`
    tic = time.perf_counter()
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(trainData, trainLabels)

    # evaluate the model and print the accuracies list
    score = model.score(valData, valLabels)
    toc = time.perf_counter()
    if score > bestScore:
        bestModel = model
        bestScore = score
    print("k=%d, accuracy=%.2f%%, finished in %.4f seconds" % (k, score * 100, toc - tic))
    accuracies.append(score)
    timings.append(toc - tic)

# largest accuracy
# np.argmax returns the indices of the maximum values along an axis
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data, it ran for %.4f seconds" % (kVals[i],
    accuracies[i] * 100, timings[i]))

j = np.argmin(timings)
print("k=%d achieved minimal running time of %.4f on validation data, it has an accuracy of %.2f%%" % (kVals[j],
    timings[j], accuracies[j] * 100))


# Predict labels for the test set
tic = time.perf_counter()
predictions = bestModel.predict(testData)
toc = time.perf_counter()
print("prediction on test set using k=%d cost %.4f seconds" % (kVals[i], toc - tic))

# Evaluate performance of model for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))
