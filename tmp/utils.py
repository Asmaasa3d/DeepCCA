import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from keras.utils.data_utils import get_file
from sklearn.model_selection import train_test_split


def load_data(data_file):
    """loads the data from the gzip pickled files, and converts to numpy arrays"""
    print('loading data ...')
    data = pd.read_csv(data_file,nrows=7271)
    """split into features and labels"""
    y = data.point
    X = data.drop('point', axis=1)
    """split into train and test"""
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
    test_set_x, valid_set_x, test_set_y, valid_set_y = train_test_split(X_test,y_test,test_size=0.5)

    train_set_x = np.asarray(X_train, dtype='float32')
    train_set_y= np.asarray(y_train, dtype='int32') 
    test_set_x = np.asarray(test_set_x, dtype='float32')
    test_set_y = np.asarray(test_set_y, dtype='int32') 
    valid_set_x=np.asarray(valid_set_x, dtype='float32')
    valid_set_y = np.asarray(valid_set_y, dtype='int32')

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]





def svm_classify(data, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, _, train_label = data[0]
    valid_data, _, valid_label = data[1]
    test_data, _, test_label = data[2]

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    return [test_acc, valid_acc]


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret

    valid_acc = accuracy_score(valid_label, p)

    return [test_acc, valid_acc]


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret

