from constants import DIR_MODEL, DIR_TEST, LABELS
from sklearn import svm
import numpy as np
import pickle
import utils
import vgg16


def test():
    X_test, y_test, paths = utils.read_process_images(
        DIR_TEST, LABELS, return_paths=True)
    print('X_test shape: %s' % str(X_test.shape))
    print('y_test shape: %s' % str(y_test.shape))

    X_test_fc6 = vgg16.extract_fc6_features(X_test, verbose=True)
    print('X_test_fc6 shape: %s' % str(X_test_fc6.shape))

    # load model from disk
    loaded_model = pickle.load(open(DIR_MODEL, 'rb'))

    # predict at test data
    y_pred = loaded_model.predict(X_test_fc6)

    print('Y test = ')  
    print(y_test)
    print('Y predict = ')
    print(y_pred)

    scores = loaded_model.score(X_test_fc6, y_test)
    print("accuracy = %f" % scores)

    # Print false predictions
    diff = y_test != y_pred
    if np.any(diff):
        print('false predictions:')
        print(paths[diff])

    # Print confidence score of each sample
    print('confidence scores:')
    print(loaded_model.decision_function(X_test_fc6))

    # Precision and Recall for motobike - 0
    tp, fp, fn = 0, 0, 0
    for p, a in zip(y_test, y_pred):
        tp += ((a == 0) and (p == 0))
        fn += ((a == 0) and (p == 1))
        fp += ((a == 1) and (p == 0))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1_score = (2 * precision * recall) / (precision + recall)
    print("Precision: {0:.3}\nRecall: {1:.3}\nF1-score: {2:.3}".format(precision, recall, f1_score))

if __name__ == '__main__':
    test()
