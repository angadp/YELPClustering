import matplotlib.pyplot as plt
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import BisectingKMeans
from pyspark import SparkContext, SparkConf
import datetime as dt
import dateutil.parser as par
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from pyspark.ml.feature import MinMaxScaler
import pyspark.ml.linalg
import sys
from pyspark.sql import SparkSession
SparkContext.setSystemProperty('spark.executor.memory', '4g')
# Find the cluster number for the points
def findCenter(vector, centroids):
    dist = sys.maxint
    cluster = -1
    i = 0
    for center in centroids:
        if(Vectors.squared_distance(center, vector)<dist):
            dist = Vectors.squared_distance(center, vector)
            cluster = i
        i += 1
    return cluster

conf = SparkConf().setAppName("test").setMaster("local[*]")
sc = SparkContext(conf=conf)

#Selecting fields and scaling using MinMaxScaler
spark = SparkSession \
    .builder \
    .appName("KMeans") \
    .config("spark.some.config.option", "Angadpreet-KMeans") \
    .getOrCreate()
today = dt.datetime.today()
spark_df = sc.parallelize(spark.read.json("Data/yelp_academic_dataset_user.json").select("review_count", "average_stars", "yelping_since").rdd.map(lambda x: (x[0], x[1], (today - par.parse(x[2])).days)).take(1700))
scaler = MinMaxScaler(inputCol="_1",\
         outputCol="scaled_1")
trial_df = spark_df.map(lambda x: pyspark.ml.linalg.Vectors.dense(x)).map(lambda x:(x, )).toDF()
scalerModel = scaler.fit(trial_df)
vector_df = scalerModel.transform(trial_df).select("scaled_1").rdd.map(lambda x:Vectors.dense(x))

#Input into the Algorithm
km = BisectingKMeans()
kme = km.train(vector_df, k = 4, maxIterations = 20, seed=2018)
centers = kme.clusterCenters

#Creating data structure with fields and cluster center
df_with = spark.createDataFrame(vector_df.map(lambda x:(float(x[0][0]), float(x[0][1]), float(x[0][2]), findCenter(x[0], centers)))).toPandas()

#Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
scatter = ax.scatter(df_with['_1'],df_with['_2'],df_with['_3'],
                     c=df_with['_4'])
ax.set_title('Bisecting K Means Clustering')
ax.set_xlabel('Review Count')
ax.set_ylabel('Average Stars')
ax.set_zlabel('Yelping since')
plt.colorbar(scatter)
plt.show()