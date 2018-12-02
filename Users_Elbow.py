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
from pyspark.ml.feature import MinMaxScaler
import pyspark.ml.linalg

conf = SparkConf().setAppName("test").setMaster("local[*]")
sc = SparkContext(conf=conf)
from pyspark.sql import SparkSession
import sys

spark = SparkSession \
    .builder \
    .appName("KMeans") \
    .config("spark.some.config.option", "Angadpreet-KMeans") \
    .getOrCreate()
today = dt.datetime.today()
spark_df = sc.parallelize(spark.read.json("Data/yelp_academic_dataset_user.json").select("review_count", "average_stars", "yelping_since").rdd.map(lambda x: (x[0], x[1])).collect()[:1700])
scaler = MinMaxScaler(inputCol="_1",\
         outputCol="scaled_1")
trial_df = spark_df.map(lambda x: pyspark.ml.linalg.Vectors.dense(x)).map(lambda x:(x, )).toDF()
scalerModel = scaler.fit(trial_df)
vector_df = scalerModel.transform(trial_df).select("scaled_1").rdd.map(lambda x:Vectors.dense(x))
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans()
    kme = km.train(vector_df, k= k, maxIterations=20, initializationMode = "random", seed=2018)
    err = vector_df.map(lambda x: (x[0], kme.predict(x[0]))).collect()

    per_clus = [0] * k
    per_clus_num = [0] *k

    # Silhoutte Value comparison
    ag = 0
    agi = 1700
    for er in err:
        avg = [0] * k
        avgi = [0] * k
        for e in err:
            avg[e[1]] += Vectors.squared_distance(er[0], e[0])
            avgi[e[1]] += 1
        a = avg[er[1]] / avgi[er[1]]
        b = sys.maxint
        for i in range(len(avg)):
            if (i != er[1] and avgi[i]!=0):
                if (avg[i] / avgi[i] < b):
                    b = avg[i] / avgi[i]
        ag += (b - a) / max(b, a)
        per_clus[er[1]] += (b - a) / max(b, a)
        per_clus_num[er[1]] += 1

    sil = (ag / agi)

    Sum_of_squared_distances.append(sil)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
