from __future__ import print_function

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix, _convert_to_vector, CoordinateMatrix, MatrixEntry
from pyspark.mllib.clustering import PowerIterationClustering
from pyspark import SparkContext, SparkConf
import datetime as dt
from pyspark.ml.feature import VectorAssembler
import dateutil.parser as par
SparkContext.setSystemProperty('spark.executor.memory', '3g')
conf = SparkConf().setAppName("test").setMaster("local")
sc = SparkContext(conf=conf)
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("KMeans") \
    .config("spark.some.config.option", "Angadpreet-KMeans") \
    .getOrCreate()
today = dt.datetime.today()
spark_df = spark.createDataFrame(spark.read.json("yelp_academic_dataset_user.json").select("review_count", "average_stars", "yelping_since").rdd.map(lambda x: (x[0], x[1], (today - par.parse(x[2])).days)).take(1700))
spark_df = spark.createDataFrame(spark_df.toPandas().transpose()).rdd
vector_df = sc.parallelize(spark_df.map(lambda s : Vectors.dense(s)).collect())
mat = RowMatrix(vector_df)
mat.rows.foreach(print)
# cm = CoordinateMatrix(
#     mat.rows.zipWithIndex().flatMap(
#         lambda x: [MatrixEntry(x[1], j, v) for j, v in enumerate(x[0])]
#     )
# ).toRowMatrix()
pre = sc.parallelize(mat.columnSimilarities().entries.map(lambda e: (e.i, e.j, e.value)).collect())
#simsPerfect = sc.parallelize(mat.columnSimilarities().entries.collect())
model = PowerIterationClustering.train(pre, 3, 10)
model.assignments().foreach(lambda x: print(str(x.id) + " -> " + str(x.cluster)))