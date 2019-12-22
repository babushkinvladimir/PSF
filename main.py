import os
import glob
# Import pandas
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from _k_medoids import KMedoids
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import scipy.optimize
from searchSequenceNumpy import searchSequenceNumpy

# import errors
from MAE import MAE
from MAPE import MAPE
from MRE import MRE
from MSE import MSE

# define results directory
directory = os.getcwd()
tmpArray = directory.split('/')[1:-1]
resDirectory = ''
for k in range(len(tmpArray) - 1):
    resDirectory = resDirectory + '/' + tmpArray[k]
resDirectory = resDirectory + '/RESULTS'

# first list all the csv files
path = os.getcwd()
os.chdir( path )

nyisoData = glob.glob( path+'/DATA/NYISO/*/**.csv' )
print(nyisoData)

# to store aggregated data
allLoadDf = pd.DataFrame(columns=['date', '1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'])
daysSkipped = []
for i in range(len(nyisoData)):
    # read csv files
    tmpDf = pd.read_csv(nyisoData[i])

    #select data for New York City only
    timeSeriesDfNYC= tmpDf.loc[tmpDf['Name'] == 'N.Y.C.']

    # this line converts the string object in Timestamp object
    dateTime = [datetime.datetime.strptime(d, "%m/%d/%Y %H:%M:%S") for d in timeSeriesDfNYC["Time Stamp"]]

    #extract only 24 hours of data (each record starting with 00 minute in timestamp)
    # this is a boolean mask, True only for timestams that contain 00 in minutes (start of the hour)
    dateTime24 = [d.minute ==0 and d.second ==0 for d in dateTime]

    #apply boolean mask to timeseries for NYC
    timeSeriesDfNyc24= timeSeriesDfNYC.loc[dateTime24]

    if len(timeSeriesDfNyc24)==24:
        # intialise data of lists.
        allLoadDf.loc[i] = [dateTime[0].date()] + list(timeSeriesDfNyc24['Load'].apply(pd.to_numeric))
    else: #skip
        daysSkipped.append(dateTime[0].date())

#rows with nan also ignored
allLoadDf= allLoadDf.dropna()



cols =['1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
dates = allLoadDf['date']

X = np.array(allLoadDf[cols].apply(pd.to_numeric))
np.argwhere(np.isnan(X))

#normalize the data
Xnorm= X/len(np.mean(X,1))

# L1 normalization ( row's unit length is one or the sum of the square of each element in a row is one)
xL1 = X / np.linalg.norm(X, axis=-1)[:, np.newaxis]

dates= dates.reset_index()
dates=dates['date']

a = allLoadDf[cols]

# plot timeseries in one figure
asOnePlot = []
asOnePlotNorm = []
asOnePlotL1 = []

for k in range(len(a)):
    asOnePlot.extend(a.iloc[k])
    asOnePlotNorm.extend(X[k])
    asOnePlotL1.extend(xL1[k])

# plot for one day, nyc
dateToPlot = datetime.datetime.strptime('2019-12-12', "%Y-%m-%d").date()
daysBeforeDate = len(dates[dates<dateToPlot])
dayToPlot = X[daysBeforeDate,:]
normDayToPlot = Xnorm[daysBeforeDate,:]
normL1DayToPlot = xL1[daysBeforeDate,:]
dateOfTheDayToPlot = dates[daysBeforeDate]

w = 12
h = 8
d = 70
plt.figure(figsize=(w, h), dpi=d)
plt.plot(dayToPlot)
plt.xticks(np.arange(0, 24, 1),fontsize=15)
plt.yticks(fontsize=15)
plt.xlim([0,23])
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Load, MW',fontsize=15)
plt.title("Load for "+str(dateOfTheDayToPlot.strftime('%d %b, %Y')),fontsize=18,y=1.03)


w = 12
h = 8
d = 70
plt.figure(figsize=(w, h), dpi=d)
plt.plot(normDayToPlot)
plt.xticks(np.arange(0, 24, 1),fontsize=15)
plt.yticks(fontsize=15)
plt.xlim([0,23])
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Load, norm. values',fontsize=15)
plt.title("Normalized load for "+str(dateOfTheDayToPlot.strftime('%d %b, %Y')),fontsize=18,y=1.03)


w = 12
h = 8
d = 70
plt.figure(figsize=(w, h), dpi=d)
plt.plot(normL1DayToPlot)
plt.xticks(np.arange(0, 24, 1),fontsize=15)
plt.yticks(fontsize=15)
plt.xlim([0,23])
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Load, norm. values',fontsize=15)
plt.title("L1 normalized load for "+str(dateOfTheDayToPlot.strftime('%d %b, %Y')),fontsize=18,y=1.03)




# plot for one month, nyc

w = 12
h = 8
d = 70
plt.figure(figsize=(w, h), dpi=d)
plt.plot(asOnePlot[0:745])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim([0,745])
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Load, MW',fontsize=15)
plt.title("Load in NYC from "+str(dates[0].strftime('%d %b, %Y'))+" to "+str(dates[30].strftime('%d %b, %Y')),fontsize=18,y=1.03)

#plot normalized for one month nyc
w = 12
h = 8
d = 70
plt.figure(figsize=(w, h), dpi=d)
plt.plot(asOnePlotNorm[0:745])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim([0,745])
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Load, normalized values',fontsize=15)
plt.title("Normalized load in NYC from "+str(dates[0].strftime('%d %b, %Y'))+" to "+str(dates[30].strftime('%d %b, %Y')),fontsize=18,y=1.03)


#plot normalized for one month nyc
w = 12
h = 8
d = 70
plt.figure(figsize=(w, h), dpi=d)
plt.plot(asOnePlotL1[0:745])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim([0,745])
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Load, normalized values',fontsize=15)
plt.title("L1 normalization for load in NYC from "+str(dates[0].strftime('%d %b, %Y'))+" to "+str(dates[30].strftime('%d %b, %Y')),fontsize=18,y=1.03)


# perform kMeans clustering
numClusters = 3
kmeans = KMeans(n_clusters=numClusters, random_state=0).fit(X)

# plot for each type of cluster
fig, axs = plt.subplots(numClusters,sharex=True,sharey=True, figsize=(8, 10))
for clust in range(numClusters):
    oneTypeOfData = a[kmeans.labels_ == clust]
    for k in range(len(oneTypeOfData)):
        axs[clust].plot(oneTypeOfData.iloc[k], alpha=0.1)
        axs[clust].set_xticks(np.arange(0, 24, 1))
        axs[clust].set_xlim([0,23])
        axs[clust].set_title('load for cluster #'+str(clust+1))
    axs[clust].plot(np.mean(oneTypeOfData), 'r--',linewidth=2, alpha=1.0)
plt.subplots_adjust(left=None, bottom=0.06, right=None, top=0.96, wspace=None, hspace=0.29)
# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Load, MW',fontsize=15, labelpad=20)


# plot a heatmap shownig how many days fall for each cluster per month
labels = kmeans.labels_
heatmapData = np.zeros((12,numClusters))
for k in range(len(dates)):
    for m in range(1,13):
        if dates[k].month == m:
            for l in range(numClusters):
                if labels[k]==l:
                    heatmapData[m-1,l]=heatmapData[m-1,l]+1

w = 8
h = 10
d = 70
plt.figure(figsize=(w, h), dpi=d)
plt.imshow(heatmapData, cmap='hot', interpolation='nearest')
yLabels = ['Jan','Feb','Mar', 'Apr','May', 'Jun','Jul','Aug','Sep','Oct','Nov','Dec']
xLabels =[str(x+1) for x in range(numClusters)]
plt.yticks(np.arange(0, 12, 1),yLabels,fontsize=15)
plt.xticks(np.arange(0, numClusters, 1),xLabels,fontsize=15)
plt.xlabel('cluster labels',fontsize=15)
plt.title("Number of days per cluster for each month, k = "+str(numClusters),fontsize=16,y=1.03)
plt.colorbar()
plt.show()


# k-MEDOIDS
numClusters = 7
kmedoids = KMedoids(n_clusters=numClusters, random_state=0).fit(X)

# plot for each type of cluster
fig, axs = plt.subplots(numClusters,sharex=True,sharey=True, figsize=(8, 10))
for clust in range(numClusters):
    oneTypeOfData = a[kmedoids.labels_ == clust]
    for k in range(len(oneTypeOfData)):
        axs[clust].plot(oneTypeOfData.iloc[k], alpha=0.1)
        axs[clust].set_xticks(np.arange(0, 24, 1))
        axs[clust].set_xlim([0,23])
        axs[clust].set_title('load for cluster #'+str(clust+1))
    axs[clust].plot(np.mean(oneTypeOfData), 'r--',linewidth=2, alpha=1.0)
plt.subplots_adjust(left=None, bottom=0.06, right=None, top=0.96, wspace=None, hspace=0.29)
# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Load, MW',fontsize=15, labelpad=20)

# plot a heatmap shownig how many days fall for each cluster per month
labels = kmedoids.labels_
heatmapData = np.zeros((12,numClusters))
for k in range(len(dates)):
    for m in range(1,13):
        if dates[k].month == m:
            for l in range(numClusters):
                if labels[k]==l:
                    heatmapData[m-1,l]=heatmapData[m-1,l]+1

w = 8
h = 10
d = 70
plt.figure(figsize=(w, h), dpi=d)
plt.imshow(heatmapData, cmap='hot', interpolation='nearest')
yLabels = ['Jan','Feb','Mar', 'Apr','May', 'Jun','Jul','Aug','Sep','Oct','Nov','Dec']
xLabels =[str(x+1) for x in range(numClusters)]
plt.yticks(np.arange(0, 12, 1),yLabels,fontsize=15)
plt.xticks(np.arange(0, numClusters, 1),xLabels,fontsize=15)
plt.xlabel('cluster labels',fontsize=15)
plt.title("Number of days per cluster for each month, k = "+str(numClusters))
plt.colorbar()
plt.show()


# HIERARCHICAL
#https://www.geeksforgeeks.org/implementing-agglomerative-clustering-using-sklearn/
numClusters = 7
labels = AgglomerativeClustering(n_clusters=numClusters, affinity='euclidean', linkage='ward').fit_predict(X)

# plot for each type of cluster
fig, axs = plt.subplots(numClusters,sharex=True,sharey=True, figsize=(8, 10))
for clust in range(numClusters):
    oneTypeOfData = a[labels == clust]
    for k in range(len(oneTypeOfData)):
        axs[clust].plot(oneTypeOfData.iloc[k], alpha=0.1)
        axs[clust].set_xticks(np.arange(0, 24, 1))
        axs[clust].set_xlim([0,23])
        axs[clust].set_title('load for cluster #'+str(clust+1))
    axs[clust].plot(np.mean(oneTypeOfData), 'r--',linewidth=2, alpha=1.0)
plt.subplots_adjust(left=None, bottom=0.06, right=None, top=0.96, wspace=None, hspace=0.29)
# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Load, MW',fontsize=15, labelpad=20)

# plot a heatmap shownig how many days fall for each cluster per month
heatmapData = np.zeros((12,numClusters))
for k in range(len(dates)):
    for m in range(1,13):
        if dates[k].month == m:
            for l in range(numClusters):
                if labels[k]==l:
                    heatmapData[m-1,l]=heatmapData[m-1,l]+1

w = 8
h = 10
d = 70
plt.figure(figsize=(w, h), dpi=d)
plt.imshow(heatmapData, cmap='hot', interpolation='nearest')
yLabels = ['Jan','Feb','Mar', 'Apr','May', 'Jun','Jul','Aug','Sep','Oct','Nov','Dec']
xLabels =[str(x+1) for x in range(numClusters)]
plt.yticks(np.arange(0, 12, 1),yLabels,fontsize=15)
plt.xticks(np.arange(0, numClusters, 1),xLabels,fontsize=15)
plt.xlabel('cluster labels',fontsize=15)
plt.title("Number of days per cluster for each month, k = "+str(numClusters))
plt.colorbar()
plt.show()



errorDfKmeans = pd.DataFrame(columns=['error','2019-12-12','2019-10-01','2019-08-08','2019-04-15'])
errorDfKmeans['error'] = ['mae','mape','mre','mse']
errorDfKmeansNorm = pd.DataFrame(columns=['error','2019-12-12','2019-10-01','2019-08-08','2019-04-15'])
errorDfKmeansNorm['error'] = ['mae','mape','mre','mse']

errorDfKmedoids = pd.DataFrame(columns=['error','2019-12-12','2019-10-01','2019-08-08','2019-04-15'])
errorDfKmedoids['error'] = ['mae','mape','mre','mse']
errorDfKmedoidsNorm = pd.DataFrame(columns=['error','2019-12-12','2019-10-01','2019-08-08','2019-04-15'])
errorDfKmedoidsNorm['error'] = ['mae','mape','mre','mse']

errorDfHierarchical = pd.DataFrame(columns=['error','2019-12-12','2019-10-01','2019-08-08','2019-04-15'])
errorDfHierarchical['error'] = ['mae','mape','mre','mse']
errorDfHierarchicalNorm = pd.DataFrame(columns=['error','2019-12-12','2019-10-01','2019-08-08','2019-04-15'])
errorDfHierarchicalNorm['error'] = ['mae','mape','mre','mse']

########################################################################################################################
# perform prediction with k-means
########################################################################################################################
import imp
import lbfPredict
imp.reload(lbfPredict)
windowSize = 5
numClusters = 3
numDaysToTrain = 365*4
method = 'k-means'
daysArray = ['2019-12-12','2019-10-01','2019-08-08','2019-04-15']
for day in daysArray:
    dateToPredict = datetime.datetime.strptime(day, "%Y-%m-%d").date()
    fileToSave = resDirectory+"/predictKMeans_"+str(numDaysToTrain)+"_"+str(dateToPredict.strftime('%d_%b_%Y'))
    # predict for non-normalized trends
    (mae,mape,mre,mse) = lbfPredict.lbfPredict(X,dates, dateToPredict, method,fileToSave,windowSize,numClusters,numDaysToTrain)
    errorDfKmeans[day] = [mae,mape,mre,mse]
    # predict for normalized trends
    (mae,mape,mre,mse) = lbfPredict.lbfPredict(X,dates, dateToPredict, method,fileToSave,windowSize,numClusters,numDaysToTrain, True)
    errorDfKmeansNorm[day] = [mae, mape, mre, mse]

errorDfKmeans.to_csv(resDirectory+'/errorkMeans.csv')
errorDfKmeansNorm.to_csv(resDirectory+'/errorkMeansNorm.csv')
########################################################################################################################
# perform prediction with k-medoids
########################################################################################################################
import imp
import lbfPredict
imp.reload(lbfPredict)
windowSize = 5
numClusters = 3
numDaysToTrain = 365*4
method = 'k-medoids'
daysArray = ['2019-12-12','2019-10-01','2019-08-08','2019-04-15']
for day in daysArray:
    dateToPredict = datetime.datetime.strptime(day, "%Y-%m-%d").date()
    fileToSave = resDirectory+"/predictKMedoids_"+str(numDaysToTrain)+"_"+str(dateToPredict.strftime('%d_%b_%Y'))
    # predict for non-normalized trends
    (mae,mape,mre,mse) = lbfPredict.lbfPredict(X,dates, dateToPredict, method,fileToSave,windowSize,numClusters,numDaysToTrain)
    errorDfKmedoids[day] = [mae, mape, mre, mse]
    # predict for normalized trends
    (mae,mape,mre,mse) = lbfPredict.lbfPredict(X,dates, dateToPredict, method,fileToSave,windowSize,numClusters,numDaysToTrain, True)
    errorDfKmedoidsNorm[day] = [mae, mape, mre, mse]

errorDfKmedoids.to_csv(resDirectory+'/errorkMedoids.csv')
errorDfKmedoidsNorm.to_csv(resDirectory+'/errorkMedoidsNorm.csv')


########################################################################################################################
# perform prediction with hierarchical
########################################################################################################################
import imp
import lbfPredict
imp.reload(lbfPredict)
windowSize = 5
numClusters = 3
numDaysToTrain = 365*4
method = 'hierarchical'
daysArray = ['2019-12-12','2019-10-01','2019-08-08','2019-04-15']
for day in daysArray:
    dateToPredict = datetime.datetime.strptime(day, "%Y-%m-%d").date()
    fileToSave = resDirectory+"/predictHierarchical_"+str(numDaysToTrain)+"_"+str(dateToPredict.strftime('%d_%b_%Y'))
    # predict for non-normalized trends
    (mae,mape,mre,mse) = lbfPredict.lbfPredict(X,dates, dateToPredict, method,fileToSave,windowSize,numClusters,numDaysToTrain)
    errorDfHierarchical[day] = [mae, mape, mre, mse]
    # predict for normalized trends
    (mae,mape,mre,mse) = lbfPredict.lbfPredict(X,dates, dateToPredict, method,fileToSave,windowSize,numClusters,numDaysToTrain, True)
    errorDfHierarchicalNorm[day] = [mae, mape, mre, mse]

errorDfHierarchical.to_csv(resDirectory+'/errorHierarchical.csv')
errorDfHierarchicalNorm.to_csv(resDirectory+'/errorHierarchicalNorm.csv')


# to store the errors:
errorDfEnsemble = pd.DataFrame(columns=['error','2019-12-12','2019-10-01','2019-08-08','2019-04-15'])
errorDfEnsemble['error'] = ['mae','mape','mre','mse']

########################################################################################################################
# ensemble model:
########################################################################################################################
numModels = 3
windowSize = 5
numPrecidedDays = 10
numClusters = 3
numDaysToTrain = 365*4

daysArray = ['2019-12-12','2019-10-01','2019-08-08','2019-04-15']
for day in daysArray:

    w = np.ones(numModels)/numModels

    weightsArray = w


    dateToPredict = datetime.datetime.strptime(day, "%Y-%m-%d").date()
    daysBeforeDate = len(dates[dates<dateToPredict])

    finalDayToTestNonNorm = X[daysBeforeDate, :]

    # we train on 11 months
    daysToTrain = xL1[daysBeforeDate-numDaysToTrain-30:daysBeforeDate-30]

    finalDayToTest = xL1[daysBeforeDate,:]
    finalDayDate = dates[daysBeforeDate]

    # we train weights on 1 month
    daysToTest = xL1[daysBeforeDate-30:daysBeforeDate,:]



    trainSet = daysToTrain

    for d in range(len(daysToTest)):

        testSet = daysToTest[d,:]

        #k-means
        kmeans = KMeans(n_clusters=numClusters, random_state=0).fit(trainSet)

        # find a label for a day before the day of interest
        dayBefore = kmeans.labels_[-1]

        # find sequence of 5 days preceding the day of interest
        sequenceToLook = kmeans.labels_[numDaysToTrain - windowSize:numDaysToTrain]

        # find the first indeces of the sequences
        indecesFound = searchSequenceNumpy(kmeans.labels_, sequenceToLook)

        arrayOfSimilarDaysIdx = []

        for k in range(len(indecesFound) - 1):
            tmp = indecesFound[k] + windowSize + 1
            # we don't want the last 5 days to appear in the subeset of similar days
            if tmp.item(0) < len(daysToTrain) - windowSize:
                arrayOfSimilarDaysIdx.append(tmp.item(0))

        similarDays = daysToTrain[arrayOfSimilarDaysIdx]

        if np.size(arrayOfSimilarDaysIdx) > 1:
            predictedDay1 = np.mean(similarDays, 0)
        else:
            if np.size(arrayOfSimilarDaysIdx) == 1:
                predictedDay1 = similarDays[0]
            else:
                predictedDay1 = np.zeros(24)

        #  kmedoids
        kmedoids = KMedoids(n_clusters=numClusters, random_state=0).fit(trainSet)

        # find a label for a day before the day of interest
        dayBefore = kmedoids.labels_[-1]

        # find sequence of 5 days preceding the day of interest
        sequenceToLook = kmedoids.labels_[numDaysToTrain - windowSize:numDaysToTrain]

        # find the first indeces of the sequences
        indecesFound = searchSequenceNumpy(kmedoids.labels_, sequenceToLook)

        arrayOfSimilarDaysIdx = []

        for k in range(len(indecesFound) - 1):
            tmp = indecesFound[k] + windowSize + 1
            # we don't want the last 5 days to appear in the subeset of similar days
            if tmp.item(0) < len(daysToTrain) - windowSize:
                arrayOfSimilarDaysIdx.append(tmp.item(0))

        similarDays = daysToTrain[arrayOfSimilarDaysIdx]

        if np.size(arrayOfSimilarDaysIdx) > 1:
            predictedDay2 = np.mean(similarDays, 0)
        else:
            if np.size(arrayOfSimilarDaysIdx) == 1:
                predictedDay2 = similarDays[0]
            else:
                predictedDay2 = np.zeros(24)

        #agglomerative clustering
        hclust = AgglomerativeClustering(n_clusters=numClusters, affinity='euclidean', linkage='ward').fit_predict(
            trainSet)

        # find a label for a day before the day of interest
        dayBefore = hclust[-1]

        # find sequence of 5 days preceding the day of interest
        sequenceToLook = hclust[numDaysToTrain - windowSize:numDaysToTrain]

        # find the first indeces of the sequences
        indecesFound = searchSequenceNumpy(hclust, sequenceToLook)

        arrayOfSimilarDaysIdx = []

        for k in range(len(indecesFound) - 1):
            tmp = indecesFound[k] + windowSize + 1
            # we don't want the last 5 days to appear in the subeset of similar days
            if tmp.item(0) < len(daysToTrain) - windowSize:
                arrayOfSimilarDaysIdx.append(tmp.item(0))

        similarDays = daysToTrain[arrayOfSimilarDaysIdx]

        if np.size(arrayOfSimilarDaysIdx) > 1:
            predictedDay3 = np.mean(similarDays, 0)
        else:
            if np.size(arrayOfSimilarDaysIdx) == 1:
                predictedDay3 = similarDays[0]
            else:
                predictedDay3 = np.zeros(24)


        if numModels == 2:
            pM = w[0]*predictedDay1 + w[1]*predictedDay2
            x=w
            banana = lambda x: 100/len(testSet)*np.sum(np.abs((predictedDay1*x[0]+predictedDay2*x[1])-testSet)/np.mean(testSet))
            xopt = scipy.optimize.minimize(banana, np.ones(numModels) / numModels, method='L-BFGS-B',
                                           bounds=((0, 1), (0, 1)))

        if numModels == 3:
            pM = w[0]*predictedDay1 + w[1]*predictedDay2 + w[2]*predictedDay3
            x=w
            banana = lambda x: 100/len(testSet)*np.sum(np.abs((predictedDay1*x[0]+predictedDay2*x[1]+predictedDay3*x[2])-testSet)/np.mean(testSet))

            xopt =scipy.optimize.minimize(banana, np.ones(numModels)/numModels, method = 'L-BFGS-B',bounds=((0, 1),(0,1),(0,1)))

        w = xopt.x

        weightsArray=np.row_stack((weightsArray,w))

        # add the day to the training set
        trainSet = np.row_stack((trainSet, testSet))

    # predict with ensamble ???
    newW = np.mean(weightsArray,0)
    print(newW)


    if numModels == 2:
        kmeans = KMeans(n_clusters=numClusters, random_state=0).fit(trainSet)
        # find a label for a day before the day of interest
        dayBefore = kmeans.labels_[-1]

        # find sequence of 5 days preceding the day of interest
        sequenceToLook = kmeans.labels_[numDaysToTrain - windowSize:numDaysToTrain]

        # find the first indeces of the sequences
        indecesFound = searchSequenceNumpy(kmeans.labels_, sequenceToLook)

        arrayOfSimilarDaysIdx = []

        for k in range(len(indecesFound) - 1):
            tmp = indecesFound[k] + windowSize + 1
            # we don't want the last 5 days to appear in the subeset of similar days
            if tmp.item(0) < len(daysToTrain) - windowSize:
                arrayOfSimilarDaysIdx.append(tmp.item(0))

        similarDays = daysToTrain[arrayOfSimilarDaysIdx]

        if np.size(arrayOfSimilarDaysIdx) > 1:
            predictedDay1 = np.mean(similarDays, 0)
        else:
            if np.size(arrayOfSimilarDaysIdx)==1:
                predictedDay2 = similarDays[0]
            else:
                predictedDay2 = np.zeros(24)

        #  kmedoids
        kmedoids = KMedoids(n_clusters=numClusters, random_state=0).fit(trainSet)

        # find a label for a day before the day of interest
        dayBefore = kmedoids.labels_[-1]

        # find sequence of 5 days preceding the day of interest
        sequenceToLook = kmedoids.labels_[numDaysToTrain - windowSize:numDaysToTrain]

        # find the first indeces of the sequences
        indecesFound = searchSequenceNumpy(kmedoids.labels_, sequenceToLook)

        arrayOfSimilarDaysIdx = []

        for k in range(len(indecesFound) - 1):
            tmp = indecesFound[k] + windowSize + 1
            # we don't want the last 5 days to appear in the subeset of similar days
            if tmp.item(0) < len(daysToTrain) - windowSize:
                arrayOfSimilarDaysIdx.append(tmp.item(0))

        similarDays = daysToTrain[arrayOfSimilarDaysIdx]

        if np.size(arrayOfSimilarDaysIdx) > 1:
            predictedDay2 = np.mean(similarDays, 0)
        else:
            if np.size(arrayOfSimilarDaysIdx) == 1:
                predictedDay2 = similarDays[0]
            else:
                predictedDay2 = np.zeros(24)
        predictedDay = newW[0] * predictedDay1 + newW[1] * predictedDay2
    if numModels == 3:
        kmeans = KMeans(n_clusters=numClusters, random_state=0).fit(trainSet)
        # find a label for a day before the day of interest
        dayBefore = kmeans.labels_[-1]

        # find sequence of 5 days preceding the day of interest
        sequenceToLook = kmeans.labels_[numDaysToTrain - windowSize:numDaysToTrain]

        # find the first indeces of the sequences
        indecesFound = searchSequenceNumpy(kmeans.labels_, sequenceToLook)

        arrayOfSimilarDaysIdx = []

        for k in range(len(indecesFound) - 1):
            tmp = indecesFound[k] + windowSize + 1
            # we don't want the last 5 days to appear in the subeset of similar days
            if tmp.item(0) < len(daysToTrain) - windowSize:
                arrayOfSimilarDaysIdx.append(tmp.item(0))

        similarDays = daysToTrain[arrayOfSimilarDaysIdx]

        if np.size(arrayOfSimilarDaysIdx) > 1:
            predictedDay1 = np.mean(similarDays, 0)
        else:
            if np.size(arrayOfSimilarDaysIdx) == 1:
                predictedDay1 = similarDays[0]
            else:
                predictedDay1 = np.zeros(24)

        #  kmedoids
        kmedoids = KMedoids(n_clusters=numClusters, random_state=0).fit(trainSet)

        # find a label for a day before the day of interest
        dayBefore = kmedoids.labels_[-1]

        # find sequence of 5 days preceding the day of interest
        sequenceToLook = kmedoids.labels_[numDaysToTrain - windowSize:numDaysToTrain]

        # find the first indeces of the sequences
        indecesFound = searchSequenceNumpy(kmedoids.labels_, sequenceToLook)

        arrayOfSimilarDaysIdx = []

        for k in range(len(indecesFound) - 1):
            tmp = indecesFound[k] + windowSize + 1
            # we don't want the last 5 days to appear in the subeset of similar days
            if tmp.item(0) < len(daysToTrain) - windowSize:
                arrayOfSimilarDaysIdx.append(tmp.item(0))

        similarDays = daysToTrain[arrayOfSimilarDaysIdx]

        if np.size(arrayOfSimilarDaysIdx) > 1:
            predictedDay2 = np.mean(similarDays, 0)
        else:
            if np.size(arrayOfSimilarDaysIdx)==1:
                predictedDay2 = similarDays[0]
            else:
                predictedDay2 = np.zeros(24)

        # agglomerative clustering
        hclust = AgglomerativeClustering(n_clusters=numClusters, affinity='euclidean', linkage='ward').fit_predict(
            trainSet)

        # find a label for a day before the day of interest
        dayBefore = hclust[-1]

        # find sequence of 5 days preceding the day of interest
        sequenceToLook = hclust[numDaysToTrain - windowSize:numDaysToTrain]

        # find the first indeces of the sequences
        indecesFound = searchSequenceNumpy(hclust, sequenceToLook)

        arrayOfSimilarDaysIdx = []

        for k in range(len(indecesFound) - 1):
            tmp = indecesFound[k] + windowSize + 1
            # we don't want the last 5 days to appear in the subeset of similar days
            if tmp.item(0) < len(daysToTrain) - windowSize:
                arrayOfSimilarDaysIdx.append(tmp.item(0))

        similarDays = daysToTrain[arrayOfSimilarDaysIdx]

        if np.size(arrayOfSimilarDaysIdx) > 1:
            predictedDay3 = np.mean(similarDays, 0)
        else:
            if np.size(arrayOfSimilarDaysIdx)==1:
                predictedDay3 = similarDays[0]
            else:
                predictedDay3 = np.zeros(24)

        predictedDay = newW[0] * predictedDay1 + newW[1] * predictedDay2+ newW[2] * predictedDay3

    # denormalize:

    scalingFactor = finalDayToTestNonNorm[0] / finalDayToTest[0]
    predictedDay=predictedDay*np.linalg.norm(predictedDay)
    predictedDay = predictedDay*scalingFactor
    finalDayToTest = finalDayToTest*scalingFactor
    fileToSave = fileToSave + "_L1norm"

    mae = MAE(finalDayToTest,predictedDay)
    mape = MAPE(finalDayToTest,predictedDay)
    mre = MRE(finalDayToTest,predictedDay)
    mse = MSE(finalDayToTest,predictedDay)
    errorDfEnsemble[day] = [mae, mape, mre, mse]
    width = 12
    height = 10
    dpi = 70
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_subplot(111)
    plt.plot(predictedDay, 'r-', label = 'predicted')
    plt.plot(finalDayToTest, label = "actual")
    plt.xticks(np.arange(0, 24, 1),fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([0,23])
    plt.xlabel('Hours',fontsize=15)
    plt.ylabel('Load, MW',fontsize=15)
    plt.title("Predicted for "+str(finalDayDate.strftime('%d %b, %Y'))+" with ensamble of "+ str(numModels)+ " models, days trained = "+str(len(trainSet)),fontsize=18,y=1.03)
    plt.legend(loc='upper left')
    ax.text(0.01, 0.87,'mae = '+str(np.round(mae,3)),transform=ax.transAxes)
    ax.text(0.01, 0.84,'mape = '+str(np.round(mape,3)),transform=ax.transAxes)
    ax.text(0.01, 0.81,'mre = '+str(np.round(mre,3)),transform=ax.transAxes)
    ax.text(0.01, 0.78,'mse = '+str(np.round(mse,3)),transform=ax.transAxes)

    fileToSave = resDirectory + "/predictEnsemble_" + str(numDaysToTrain) + "_" + str(finalDayDate.strftime('%d_%b_%Y')+"_norm")

    fig.savefig(fileToSave, bbox_inches='tight',)



errorDfEnsemble.to_csv(resDirectory+'/errorEnsemble.csv')