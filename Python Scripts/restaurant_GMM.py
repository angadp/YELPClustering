from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import GaussianMixture, KMeans
from pyspark import SparkContext, SparkConf
import datetime as dt
import dateutil.parser as par
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from pyspark.ml.feature import StandardScaler
import pyspark.ml.linalg
import sys
import pandas

conf = SparkConf().setAppName("test").setMaster("local[*]")
sc = SparkContext(conf=conf)
from pyspark.sql import SparkSession

def func1(x, y):
    z = x[0] + y[0]
    s = x[1] + y[1]
    return (z, s)

spark = SparkSession \
    .builder \
    .appName("KMeans") \
    .config("spark.some.config.option", "Angadpreet-KMeans") \
    .getOrCreate()
today = dt.datetime.today()
spark_df = sc.parallelize(spark.read.json("Data/yelp_academic_dataset_business.json").select("stars","review_count","is_open").take(1700))
scaler = StandardScaler(inputCol="_1",\
         outputCol="scaled_1")
# Getting the input data
trial_df = spark_df.map(lambda x: pyspark.ml.linalg.Vectors.dense(x)).map(lambda x:(x, )).toDF()
scalerModel = scaler.fit(trial_df)
vector_df = scalerModel.transform(trial_df).select("scaled_1").rdd.map(lambda x:Vectors.dense(x))
# from pyspark.mllib.util import MLUtils
# df = MLUtils.convertVectorColumnsFromML(df, "features")
# vector_df = scalerModel.transform(trial_df).select("scaled_1").rdd
# vector_df1 = vector_df.map(lambda x:pyspark.ml.linalg.Vectors.dense(float(x[0][0]), float(x[0][1]), float(x[0][2])))
# trial_df = vector_df1.map(lambda x: pyspark.ml.linalg.Vectors.dense(x)).map(lambda x:(x, )).toDF(["features"])
# vector_df2 = vector_df.map(lambda x:Vectors.dense(x))
#
# # Initialize K Means
km = KMeans()
#kme = km.train(vector_df, k = 4, maxIterations = 20, initializationMode = "random", seed=2018)
# Initialize GMM
gmm = GaussianMixture.train(vector_df, k=3, maxIterations=240, seed=2018)
#model = gmm.fit(trial_df)
#sc.broadcast(gmm)
#transformed_df = gmm.transform(trial_df)  # assign data to gaussian components ("clusters")
df = pandas.DataFrame({'features':[], 'cluster':[]})
i = 0
for v in vector_df.collect():
    df.loc[i] = [[float(v[0]), float(v[1]), float(v[2])], int(gmm.predict(v))]
    i+=1

print df

err = spark.createDataFrame(df).rdd.map(lambda x:(x[0], int(x[1]))).collect()
num_clusters = 3

per_clus = [0]*num_clusters
per_clus_num = [0]*num_clusters

#Silhoutte Value comparison
ag = 0
agi = 1700
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

df_with = spark.createDataFrame(spark.createDataFrame(df).rdd.map(lambda x:(x[0][0], x[0][1], x[0][2], int(x[1])))).toPandas()
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
scatter = ax.scatter(df_with['_1'],df_with['_2'],df_with['_3'],
                     c=df_with['_4'])
ax.set_title('GMM Clustering')
ax.set_xlabel('Review Count')
ax.set_ylabel('Average Stars')
ax.set_zlabel('Yelping Since')
plt.colorbar(scatter)
plt.show()