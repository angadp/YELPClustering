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
hun = model.assignments().map(lambda x: (x.cluster, (Vectors.dense(bun[0][x.id], bun[1][x.id], bun[2][x.id]), 1)))
centroids = hun.reduceByKey(lambda x, y: func1(x, y)).map(lambda x: (x[0], x[1][0]/x[1][1])).collect()
error_df = hun.map(lambda x:(Vectors.squared_distance(x[1][0], centroids[x[0]][1]))).reduce(lambda a, b: a+b)
print(error_df)