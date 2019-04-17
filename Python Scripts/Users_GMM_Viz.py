from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import GaussianMixture, KMeans
from pyspark import SparkContext, SparkConf
import datetime as dt
import dateutil.parser as par
from pyspark.ml.feature import MinMaxScaler
import pyspark.ml.linalg
import sys
import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

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
spark_df = sc.parallelize(spark.read.json("Data/yelp_academic_dataset_user.json").select("review_count", "average_stars", "yelping_since").rdd.map(lambda x: (x[0], x[1], (today - par.parse(x[2])).days)).collect()[:1700])
scaler = MinMaxScaler(inputCol="_1",\
         outputCol="scaled_1")
# Getting the input data
trial_df = spark_df.map(lambda x: pyspark.ml.linalg.Vectors.dense(x)).map(lambda x:(x, )).toDF()
scalerModel = scaler.fit(trial_df)
vector_df = scalerModel.transform(trial_df).select("scaled_1").rdd.map(lambda x:Vectors.dense(x))

# Initialize GMM
gmm = GaussianMixture.train(vector_df, k=4, maxIterations=20, seed=2018)

df = pandas.DataFrame({'features':[], 'cluster':[]})
i = 0
for v in vector_df.collect():
    df.loc[i] = [[float(v[0]), float(v[1]), float(v[2])], int(gmm.predict(v))]
    i+=1

print df

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