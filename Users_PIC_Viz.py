from __future__ import print_function

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix, _convert_to_vector, CoordinateMatrix, MatrixEntry
from pyspark.mllib.clustering import PowerIterationClustering
from pyspark import SparkContext, SparkConf
import datetime as dt
from pyspark.ml.feature import VectorAssembler
import dateutil.parser as par
SparkContext.setSystemProperty('spark.executor.memory', '4g')
conf = SparkConf().setAppName("test").setMaster("local")
sc = SparkContext(conf=conf)
from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler
import pyspark.ml.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
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
spark_df = sc.parallelize(spark.read.json("Data/yelp_academic_dataset_user.json").select("review_count", "average_stars", "yelping_since").rdd.map(lambda x: (x[0], x[1], (today - par.parse(x[2])).days)).take(1700))
scaler = MinMaxScaler(inputCol="_1",\
         outputCol="scaled_1")
trial_df = spark_df.map(lambda x: pyspark.ml.linalg.Vectors.dense(x)).map(lambda x:(x, )).toDF()
scalerModel = scaler.fit(trial_df)
vec_df = spark.createDataFrame(scalerModel.transform(trial_df).select("scaled_1").rdd.map(lambda x:(float(x[0][0]), float(x[0][1]), float(x[0][2]))))
spark_df = spark.createDataFrame(vec_df.toPandas().transpose()).rdd
vector_df = sc.parallelize(spark_df.map(lambda s : Vectors.dense(s)).collect())
mat = RowMatrix(vector_df)
bun = mat.rows.collect()
# cm = CoordinateMatrix(
#     mat.rows.zipWithIndex().flatMap(
#         lambda x: [MatrixEntry(x[1], j, v) for j, v in enumerate(x[0])]
#     )
# ).toRowMatrix()
pre = sc.parallelize(mat.columnSimilarities().entries.map(lambda e: (e.i, e.j, e.value)).collect())
#simsPerfect = sc.parallelize(mat.columnSimilarities().entries.collect())
model = PowerIterationClustering.train(pre, 3, 40, "random")
df_with = spark.createDataFrame(model.assignments().map(lambda x: (float(bun[0][x.id]), float(bun[1][x.id]), float(bun[2][x.id]), x.cluster))).toPandas()
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
scatter = ax.scatter(df_with['_1'],df_with['_2'],df_with['_3'],
                     c=df_with['_4'])
ax.set_title('Spectral/Power Iteration Clustering')
ax.set_xlabel('Review Count')
ax.set_ylabel('Average Stars')
plt.colorbar(scatter)
plt.show()