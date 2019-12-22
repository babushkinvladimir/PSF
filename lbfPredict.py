import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from _k_medoids import KMedoids
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from searchSequenceNumpy import searchSequenceNumpy
# import errors
from MAE import MAE
from MAPE import MAPE
from MRE import MRE
from MSE import MSE

def lbfPredict(X,dates,dateToPredict, method, fileToSave,windowSize=5,numClusters=3,numDaysToTrain=365, normalize = False):

    if normalize:
        daysBeforeDate = len(dates[dates < dateToPredict])
        finalDayToTestNonNorm = X[daysBeforeDate, :]
        X = X / np.linalg.norm(X, axis=-1)[:, np.newaxis]

    daysBeforeDate = len(dates[dates < dateToPredict])

    # we train on 11 months
    daysToTrain = X[daysBeforeDate - numDaysToTrain:daysBeforeDate]

    finalDayToTest = X[daysBeforeDate, :]
    finalDayDate = dates[daysBeforeDate]

    if method == 'k-means':
        # train kmeans
        model = KMeans(n_clusters=numClusters, random_state=0).fit(daysToTrain)
        labels = model.labels_
    if method == 'k-medoids':
        # train kmeans
        model = KMedoids(n_clusters=numClusters, random_state=0).fit(daysToTrain)
        labels = model.labels_
    if method == 'hierarchical':
        labels = AgglomerativeClustering(n_clusters=numClusters, affinity='euclidean', linkage='ward').fit_predict(daysToTrain)

    # find a label for a day before the day of interest
    dayBefore = labels[-1]

    # find sequence of 5 days preceding the day of interest
    sequenceToLook = labels[numDaysToTrain - windowSize:numDaysToTrain]

    # find the first indeces of the sequences
    indecesFound = searchSequenceNumpy(labels, sequenceToLook)

    while np.size(indecesFound) == 0:
        windowSize = windowSize - 1
        # find sequence of 5 days preceding the day of interest
        sequenceToLook = labels[numDaysToTrain - windowSize:numDaysToTrain]

        # find the first indeces of the sequences
        indecesFound = searchSequenceNumpy(labels, sequenceToLook)

    arrayOfSimilarDaysIdx = []

    for k in range(len(indecesFound) - 1):
        tmp = indecesFound[k] + windowSize + 1
        # we don't want the last 5 days to appear in the subeset of similar days
        if tmp.item(0) < len(daysToTrain) - windowSize:
            arrayOfSimilarDaysIdx.append(tmp.item(0))

    similarDays = daysToTrain[arrayOfSimilarDaysIdx]

    if np.size(arrayOfSimilarDaysIdx) > 1:
        predictedDay = np.mean(similarDays, 0)
    else:
        predictedDay = similarDays[0]


    # denormalize:
    if normalize:
        scalingFactor = finalDayToTestNonNorm[0] / finalDayToTest[0]
        predictedDay=predictedDay*np.linalg.norm(predictedDay)
        predictedDay = predictedDay*scalingFactor
        finalDayToTest = finalDayToTest*scalingFactor
        fileToSave = fileToSave + "_L1norm"


    mae = MAE(finalDayToTest, predictedDay)
    mape = MAPE(finalDayToTest, predictedDay)
    mre = MRE(finalDayToTest, predictedDay)
    mse = MSE(finalDayToTest, predictedDay)

    # plot for one day, nyc
    w = 10
    h = 10
    d = 70
    fig = plt.figure(figsize=(w, h), dpi=d)
    ax = fig.add_subplot(111)
    plt.plot(predictedDay, 'r-', label='predicted')
    plt.plot(finalDayToTest, label="actual")
    plt.xticks(np.arange(0, 24, 1), fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([0, 23])
    plt.xlabel('Hours', fontsize=15)
    plt.ylabel('Load, MW', fontsize=15)
    plt.title("Predicted for " + str(finalDayDate.strftime('%d %b, %Y')) + ", "+method+", k = " + str(
        numClusters) + ", days trained = " + str(len(daysToTrain)) + ", days avg = " + str(
        np.size(arrayOfSimilarDaysIdx)), fontsize=16, y=1.03)
    plt.legend(loc='upper left')
    ax.text(0.02, 0.87, 'mae = ' + str(np.round(mae, 3)), transform=ax.transAxes)
    ax.text(0.02, 0.84, 'mape = ' + str(np.round(mape, 3)), transform=ax.transAxes)
    ax.text(0.02, 0.81, 'mre = ' + str(np.round(mre, 3)), transform=ax.transAxes)
    ax.text(0.02, 0.78, 'mse = ' + str(np.round(mse, 3)), transform=ax.transAxes)

    fig.savefig(fileToSave, bbox_inches='tight',)
    return (mae, mape,mre,mse)

