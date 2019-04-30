from __future__ import print_function
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix, _convert_to_vector, CoordinateMatrix, MatrixEntry
from pyspark.mllib.clustering import PowerIterationClustering
from pyspark import SparkContext, SparkConf
import datetime as dt
from pyspark.ml.feature import VectorAssembler
import dateutil.parser as par
import sys

SparkContext.setSystemProperty('spark.executor.memory', '4g')
conf = SparkConf().setAppName("test").setMaster("local")
sc = SparkContext(conf=conf)

from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler
import pyspark.ml.linalg

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

# Getting the data structure and scaling
spark_df = sc.parallelize(spark.read.json("Data/yelp_academic_dataset_user.json").select("review_count", "average_stars", "yelping_since").rdd.map(lambda x: (x[0], x[1], (today - par.parse(x[2])).days)).collect()[:1200])
scaler = MinMaxScaler(inputCol="_1",\
         outputCol="scaled_1")
trial_df = spark_df.map(lambda x: pyspark.ml.linalg.Vectors.dense(x)).map(lambda x:(x, )).toDF()
scalerModel = scaler.fit(trial_df)
vec_df = spark.createDataFrame(scalerModel.transform(trial_df).select("scaled_1").rdd.map(lambda x:(float(x[0][0]), float(x[0][1]), float(x[0][2]))))

# Create RowMatrix from the transpose of
spark_df = spark.createDataFrame(vec_df.toPandas().transpose()).rdd
vector_df = sc.parallelize(spark_df.map(lambda s : Vectors.dense(s)).collect())
mat = RowMatrix(vector_df)
bun = mat.rows.collect()
num_clusters = 4

pre = sc.parallelize(mat.columnSimilarities().entries.map(lambda e: (e.i, e.j, e.value)).collect())

model = PowerIterationClustering.train(pre, 4, 10, "random")
err = model.assignments().map(lambda x: (Vectors.dense(bun[0][x.id], bun[1][x.id], bun[2][x.id]), x.cluster)).collect()

per_clus = [0]*num_clusters
per_clus_num = [0]*num_clusters

# Silhoutte value
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
    per_clus[er[1]] += (b - a) / max(b, a)
    per_clus_num[er[1]] += 1

sil = (ag / agi)

print(sil)

# Number of points in each cluster
df_with = sc.parallelize(err).map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b).collect()
for ji in df_with:
    print(ji)

for i in range(len(per_clus)):
    print(per_clus[i] / per_clus_num[i])