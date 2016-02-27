#!/usr/bin/python

# Dependence [https://github.com/udacity/ud120-projects]

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

t0 = time()
classifier.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
prediction = classifier.predict(features_test)
print "training time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, prediction)
print(accuracy)
#########################################################
