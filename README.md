# ML-assignments
 Repository for submitting ML assignments.  
 This repository is created mostly for submitting my ML assignments, but also in the hope that it might help someone.(_though according to license I do not guarantee any merchantability or fitness for any particular purpose_)
## List of Files
 Datasets are not included in the repository.
* knn.py: A very crude implementation of kNN(k Nearest Neighbors) on the MNIST dataset. Its algorithm is slow and lacks robustness, but meets the assignment needs. You'll need a [converters.py](ML-project/converters.py) file to import the dataset.
* bayes.py: A simple Naive Bayes script used to classify emails. It uses the dataset provided by the book _[Machine Learning in Action](https://www.manning.com/books/machine-learning-in-action)_. The code is also largely based on the demo code provided by the book, thus also very crude. But again, it meets the requirement.
* ML-project: This folder contains code used in the final project of the course. For more, please read its [README.md](ML-project/README.md) in that folder.
## Requirements
 If you would like to actually run the scripts, here is some information you may need.  
* All of the scripts require Python 3 and NumPy to work.
* Scripts in the final project uses [scikit-learn](https://scikit-learn.org).
* IPython notebooks also require Matplotlib to display the images.
* The notebook used for demostration requires `ipywidgets` and `ipycanvas` for interactive widgets.