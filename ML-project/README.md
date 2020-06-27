# Performance comparisons of classifiers on the MNIST dataset
 This folder contains code for the final project of the course.
## Brief Introduction
 The project performs these machine learning algorithms on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
* kNN (k Nearest Neighbors)
* SVM (Support Vector Machine)
* Naive Bayes
* Logistic Regression

 All of these are tested using `scikit-learn` on Python 3.6.9 on WSL (Windows Subsystem for Linux).
## Files
 All of the machine learning algorithms have a file of its name, so it shouldn't be any trouble finding them.  
The `converters.py` file is stripped from [idx2numpy](https://github.com/ivanyu/idx2numpy). If you don't want to use the file, just import the `convert_from_file` and `convert_to_file` functions from `idx2numpy`.  
 The `deskew.py` file is used to preprocess the images for some tests. It is based on the description [here](https://fsix.github.io/mnist/Deskewing.html).  
 The best model found in the project is included in the repository for your convenience, it will not be updated.  
 Additionally, a PowerPoint presentation (_in Chinese_) is included to explain the efforts done here.