import numpy as np
import fileio
import pylab as pl
import os
from scipy import signal
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
import datetime
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import roc_curve, auc


def getData(dir_path, type):
    """fetches whale detection data from audio files in directory specified"""

    file_format = type + "%d.aiff"
    data_dir = dir_path + type +  '/'

    if type == 'train':
        labels = fileio.read_csv(data_dir + "%(type)s.csv" % {"type": type})[1]
        y = [x[1] for x in labels]
    elif type == 'test':
        y = []
        labels = ([f for f in os.listdir(data_dir)
                   if os.path.isfile(os.path.join(data_dir, f))])
    X = []
    for file_id in range(1, len(labels) + 1):
        file_name = file_format % file_id
        print(data_dir + file_name)
        X.append(fileio.readAIFF(data_dir + file_name))

    return np.array(y, dtype=np.float32), np.array(X, dtype=np.float32)


def getSpectrogramFeatures(X):
    """returns spectrogram features"""
    specFeatures = []
    for i, x in enumerate(X):
        print "...making spectrogram features for instance: %d ...." % (i + 1)
        specgramx = pl.specgram(x)[0]
        filteredX = signal.wiener(specgramx)
        specFeatures.append(filteredX.reshape(filteredX.size))
    pl.close()
    return np.array(specFeatures)


def plotROC(predicted_values, true_values):
    """plots ROC curve """
    fpr, tpr, thresholds = roc_curve(true_values, predicted_values)
    roc_auc_val = auc(fpr, tpr)
    label = 'ROC Validation Set'
    label += ' (area = %0.5f)' % roc_auc_val
    pl.plot(fpr, tpr, lw=1, label=label)
    pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic curve')
    pl.legend(loc="lower right")
    pl.show()
    pl.close()


def main():

    #specify your base directory here!
    dir_path = "/Users/tahirh/Downloads/whale_data/"

    train_y, train_X = getData(dir_path=dir_path, type='train')
    train_X = getSpectrogramFeatures(train_X)

    pred_y, pred_X = getData(dir_path=dir_path, type='test')
    pred_X = getSpectrogramFeatures(pred_X)

    #standardising data
    scaler = preprocessing.StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    pred_X = scaler.transform(pred_X)

    #split data into 70% training and 30% validation data
    split_sets = ShuffleSplit(len(train_y), n_iter=1, test_size=0.30, random_state=0)
    for train_fold, val_fold in split_sets:
        input_X, validation_X, input_y, validation_y = train_X[train_fold], train_X[val_fold], train_y[train_fold], train_y[val_fold]

    params = {'n_estimators': 500, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 1}
    clf_gradboost = GradientBoostingClassifier(verbose=1,  **params)
    val_y_pred = (clf_gradboost.fit(input_X, input_y).predict_proba(validation_X))[:, 1]
    plotROC(predicted_values=val_y_pred, true_values=validation_y)

    # select only priority features
    importances = clf_gradboost.feature_importances_
    priority_features = importances > np.mean(importances)
    new_train_X = train_X[:, priority_features]
    new_pred_X = pred_X[:, priority_features]

    # train Gradient Boosting Classifier on priority features
    params_final = {'n_estimators': 2000, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 1}
    clf_gradboost_final = GradientBoostingClassifier(verbose=1,  **params_final)
    clf_gradboost_final.fit(new_train_X, train_y)

    # save final predictions for submission
    final_pred = (clf_gradboost_final.predict_proba(new_pred_X))[:, 1]
    np.savetxt("submit-%s.csv" % (datetime.datetime.now().strftime("%d-%Y-%I:%M%p")), final_pred)


if __name__ == "__main__":
    main()
