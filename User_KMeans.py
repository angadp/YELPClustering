import matplotlib.pyplot as plt
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import KMeans
from pyspark import SparkContext, SparkConf
import datetime as dt
import dateutil.parser as par
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import sys
from pyspark.ml.feature import MinMaxScaler
import pyspark.ml.linalg
from timeit import default_timer as timer



conf = SparkConf().setAppName("test").setMaster("local[*]")
sc = SparkContext(conf=conf)
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("KMeans") \
    .config("spark.some.config.option", "Angadpreet-KMeans") \
    .getOrCreate()
today = dt.datetime.today()

# Getting the data
spark_df = sc.parallelize(spark.read.json("Data/yelp_academic_dataset_user.json").select("review_count", "average_stars", "yelping_since").rdd.map(lambda x: (x[0],x[1],  (today - par.parse(x[2])).days)).collect()[:1200])
scaler = MinMaxScaler(inputCol="_1",\
         outputCol="scaled_1")
trial_df = spark_df.map(lambda x: pyspark.ml.linalg.Vectors.dense(x)).map(lambda x:(x, )).toDF()
scalerModel = scaler.fit(trial_df)
vector_df = scalerModel.transform(trial_df).select("scaled_1").rdd.map(lambda x:Vectors.dense(x))

# Initialize K Means
km = KMeans()
start = timer()
kme = km.train(vector_df, k = 4, maxIterations = 20, initializationMode = "random", seed=2018)
end = timer()
print(end - start)
print(kme.computeCost(vector_df))
centroids = kme.clusterCenters
num_clusters = 4
err = vector_df.map(lambda x:(x[0], kme.predict(x[0]))).collect()

per_clus = [0]*num_clusters
per_clus_num = [0]*num_clusters



#Silhoutte Value comparison
ag = 0
agi = 1200
for er in err:
    avg = [0] * num_clusters
    avgi = [0] * num_clusters
    for e in err:
        avg[e[1]] += Vectors.squared_distance(er[0], e[0])
        avgi[e[1]] += 1
    a = avg[er[1]] / avgi[er[1]]
    b = sys.maxint
    for i in range(len(avg)):
        if (i != er[1]):
            if (avg[i] / avgi[i] < b):
                b = avg[i] / avgi[i]
    ag += (b - a)/max(b, a)
    per_clus[er[1]] += (b - a)/max(b, a)
    per_clus_num[er[1]] += 1

sil = (ag/agi)

print(sil)

# Number of points in each cluster
df_with = sc.parallelize(err).map(lambda x:(x[1], 1)).reduceByKey(lambda a, b: a+b).collect()
for ji in df_with:
    print(ji)

for i in range(len(per_clus)):
    print(per_clus[i]/per_clus_num[i])