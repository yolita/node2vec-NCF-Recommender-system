import pandas as pd
import numpy as np
import csv
import networkx as nx

def readFile(fileName,edgelistFilename):
    fileName.reset_index(drop=True, inplace=True)
    addUserSuffix = lambda x: "User_" + str(x)
    addMovieSuffix = lambda x: "Item_" + str(x)
    csvFile = fileName #pd.read_csv(fileName)

    # columnName=csvFile.columns
    # if ('itemId' in columnName)==True:
    #     csvFile.columns = ['userId', 'movieId', 'rating', 'timestamp']

    csvFile['userId']=csvFile['userId'].apply(addUserSuffix)
    csvFile['itemId'] = csvFile['itemId'].apply(addMovieSuffix)
    useritem_EdgesFile=csvFile[['userId','itemId']].copy()
    print(useritem_EdgesFile.head())

    Graph =nx.Graph()
    G = nx.from_pandas_edgelist(useritem_EdgesFile, source='userId', target='itemId', create_using=Graph)
    nx.write_edgelist(G, edgelistFilename, encoding='utf-8')

def ratingCount(dataset):
    UserRatingCount_20m = dataset.groupby('userId')[['rating']].count()
    return UserRatingCount_20m

def partition20mto10m(file, n):
    dataset20m=file
    dataset10m_1 =dataset20m.sample(frac=n,replace=False, random_state=5)
    dataset10m_2 = dataset20m.drop(dataset10m_1.index)
    return dataset10m_1

def process_movielens_1m():
    data_frame = pd.read_csv("movielens_ratings_1m.csv", sep="::", usecols=[0, 1, 2],
                             names=['userId', 'itemId', 'rating'],
                             engine='python')
    print(data_frame.head())


# loc_movielens_100k = "/Users/yolandegoswell/Desktop/ratings.csv"
# loc_movielens_20m = "/Users/yolandegoswell/Desktop/movielens-20m-dataset/rating.csv"
# loc_movielens_10m_1 = "movielens_10m_1.csv"
# loc_movielens_1m = "ratings_1m.csv"
# loc_amazon_musical="Amazon_Musical_Instruments.csv"


# d1,d2=partition20mto10m(loc_movielens_20m)
# d1.to_csv("movielens_10m_1.csv",  encoding='utf-8')
# d2.to_csv("movielens_10m_2.csv",  encoding='utf-8')

#print(d1.describe())
#print(d2.describe())
#print(dff.head())

#dff=pd.read_csv(loc_amazon_musical,usecols=['userId','itemId','rating','timestamp'])
#print(dff.head())
#dff.drop(usecols=['userId','itemId','rating','timestamp'],inplace=True)
#print(dff.head())
#d2=partition20mto10m(dff,0.5)
#print(d2.describe())
df2=pd.read_csv("movielens_ratings_1m.csv")
#df2.columns=["userId","itemId","rating"]
#df2.to_csv("dating_ratings.csv", index=None)
readFile(df2, "movielens_1m.edgelist")

# ratings_Musical_Instruments = pd.read_csv("/Users/yolandegoswell/Desktop/ratings_Musical_Instruments.csv")
# ratings_Musical_Instruments.columns = ['userId', 'itemId', 'rating', 'timestamp']
# ratings_Musical_Instruments.to_csv('Amazon_Musical_Instruments.csv')

# dt=pd.read_csv('movielens_1m.edgelist')
# print(dt.shape)