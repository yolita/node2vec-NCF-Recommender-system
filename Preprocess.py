import pandas as pd
import numpy as np
np.random.seed(4)
from tensorflow import keras
from sklearn.model_selection import train_test_split, LeaveOneOut


def seperateIDs(val):
    if val.startswith('User_'):
        return 1
    elif val.startswith('Item_'):
        return 2

def removePrefix(str):
    prefix, id = str.split("_",1)
    return int(id)
    #return id


def ratingsReconstruct(uEmbed, mEmbed, rating):
    print(rating['userId'].dtypes)
    print(rating['itemId'].dtypes)
    print(uEmbed['userId'].dtypes)
    print(mEmbed['itemId'].dtypes)

    RatingEmbed = rating.merge(uEmbed, on="userId", how="inner", )
    RatingEmbed = RatingEmbed.merge(mEmbed, on="itemId", how="inner", suffixes=('_u', '_i'))
    return RatingEmbed


def seperateEntities(dataset):
    userDataset = dataset.iloc[:, lambda x: x.columns.str.endswith("u")]
    movieDataset = dataset.iloc[:, lambda x: x.columns.str.endswith("i")]
    #print("Entity")
    return [userDataset, movieDataset]


def splitDataset(sampleData,Target,size):
    sampleDataTrain, sampleDataTest,  labelTrain, labelTest = train_test_split(sampleData, Target, test_size=size, random_state=5)
    return [sampleDataTrain, sampleDataTest, labelTrain, labelTest]

def leave_n_outSplit(sampleData,size):
    testSet = sampleData.groupby('userId',group_keys=False).apply(lambda df: df.sample(n=size,replace=False,random_state=4))
    trainSet = sampleData.drop(testSet.index)
    trainSetLabel = trainSet['rating']
    # testSetLabel = testSet['rating']
    return [trainSet,trainSetLabel,testSet]

def customSplitWithValset(sampleData, testSize, valSize):
    '''testSize is percentage of rating sets per user total rated items in dataset
        valsize is number of samples per userId'''
    testSet = sampleData.groupby('userId', group_keys=False).apply(lambda df: df.sample(frac=testSize, replace=False, random_state=4))
    trainSet = sampleData.drop(testSet.index)
    validationSet = trainSet.groupby('userId', group_keys=False).apply(lambda df: df.sample(n=valSize, replace=False,random_state=4))
    trainSet = sampleData.drop(validationSet.index)
    trainSetLabel = trainSet['rating']
    testSetLabel = testSet['rating']
    valSetLabel = validationSet['rating']

    return [trainSet, validationSet, testSet, trainSetLabel, valSetLabel, testSetLabel]

def timeSplit(sampleData, n):
    testSet = sampleData.groupby('userId', group_keys=False).apply(lambda df: df.sort_values(by=['timestamp'], ascending=False).head(n))
    #print(testSet.shape)
    #print(testSet.head(5))
    trainSet = sampleData.drop(testSet.index)
    Label=trainSet['rating']
    #print(trainSet.head())
    trainSet.drop(columns=['rating'], inplace=True)
    #print(trainSet.head())
    #trainSetLabel = trainSet['rating']
    #testSetLabel = testSet['rating']
   # print(trainSet.shape)
    return [trainSet,Label, testSet]

def processEmbeddingData(EmbedFile):
    EmbedFile['indicator'] = EmbedFile[0].apply(seperateIDs)
    userEmbedding = EmbedFile[EmbedFile['indicator'] == 1].copy()
    movieEmbedding = EmbedFile[EmbedFile['indicator'] == 2].copy()
    userEmbedding['userId'] = userEmbedding[0].apply(removePrefix)
    movieEmbedding['itemId'] = movieEmbedding[0].apply(removePrefix)
    userEmbedding.drop(columns=["indicator", 0], inplace=True)
    movieEmbedding.drop(columns=["indicator", 0], inplace=True)

    return [userEmbedding,movieEmbedding]

def getTarget(Dataset):
    Target = Dataset['rating']
    return Target


def customDeepCoNNSplit(Dataset, test_size, userEmbedding, movieEmbedding):
    '''1. get all unique reviewers
       2. extract the unique reviewers' data points to a test set
       3. place all remaining data points in a training set
       4. from the test set get a set of unique movies
       5. remove all entries with these movies from the training set

       Dataset is the initial rating dataset
       userEmbedding is the embedding values generated through node2vec for users
       movieEmbedding is the embedding values generated through node2vec for movies
    '''
    #test_size = testNo
    # get test_size percentage of users
    unique_users = Dataset.loc[:,"userId"].unique()
    users_size = len(unique_users)
    test_idx = np.random.choice(users_size, size=int(users_size * test_size), replace=False)
    # get test users
    test_users = unique_users[test_idx]
    # everyone else is a training user
    train_users = np.delete(unique_users, test_idx)
    test = Dataset[Dataset["userId"].isin(test_users)]
    train = Dataset[Dataset["userId"].isin(train_users)]
    unique_test_movies = test["itemId"].unique()
    # drop the movies that also appear in our test set. In order to be
    # a true train/test split, we are forced to discard some data entirely
    train = train.where(np.logical_not(train["itemId"].isin(unique_test_movies))).dropna()
    train = ratingsReconstruct(userEmbedding, movieEmbedding,train)
    test = ratingsReconstruct(userEmbedding, movieEmbedding,test)
    labelTrain = train["rating"]
    labelTest = test["rating"]

    return train, test, labelTrain, labelTest
