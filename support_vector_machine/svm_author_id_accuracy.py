#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

## These lines slice the training dataset down to 1% of its original
## size, tossing out 99% of the training data.
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]


def calculateSVCAccuracy(controlTradeOff):

    from sklearn.svm import SVC
    classifier = SVC(kernel="rbf", C=controlTradeOff)

    t0 = time()
    classifier.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"

    t0 = time()
    predictions = classifier.predict(features_test)
    print "training time:", round(time()-t0, 3), "s"

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(predictions, labels_test)
    return accuracy;

#########################################################
### Main ###

print "Control Trade Off = 10 -> accuracy = %s" % calculateSVCAccuracy(10)
print "Control Trade Off = 100 -> accuracy = %s" % calculateSVCAccuracy(100)
print "Control Trade Off = 1000 -> accuracy = %s" % calculateSVCAccuracy(1000)
print "Control Trade Off = 10000 -> accuracy = %s" % calculateSVCAccuracy(10000)

#########################################################
