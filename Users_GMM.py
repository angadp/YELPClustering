from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import GaussianMixture
from pyspark import SparkContext, SparkConf
import datetime as dt
import dateutil.parser as par
from pyspark.ml.feature import MinMaxScaler
import pyspark.ml.linalg

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
spark_df = sc.parallelize(spark.read.json("Data/yelp_academic_dataset_user.json").select("review_count", "average_stars", "yelping_since").rdd.map(lambda x: (x[0], x[1], (today - par.parse(x[2])).days)).take(1700))
scaler = MinMaxScaler(inputCol="_1",\
         outputCol="scaled_1")
trial_df = spark_df.map(lambda x: pyspark.ml.linalg.Vectors.dense(x)).map(lambda x:(x, )).toDF()
scalerModel = scaler.fit(trial_df)
vector_df = scalerModel.transform(trial_df).select("scaled_1").rdd.map(lambda x:pyspark.ml.linalg.Vectors.dense(float(x[0][0]), float(x[0][1]), float(x[0][2])))
trial_df = vector_df.map(lambda x: pyspark.ml.linalg.Vectors.dense(x)).map(lambda x:(x, )).toDF(["features"])
gmm = GaussianMixture(k=3)
model = gmm.fit(trial_df)

transformed_df = model.transform(trial_df)  # assign data to gaussian components ("clusters")
final_df = transformed_df.rdd.map(lambda x:(x[1], (x[0], 1)))
centroids = final_df.reduceByKey(lambda x, y: func1(x, y)).map(lambda x: (x[0], x[1][0]/x[1][1])).collect()
error_df = transformed_df.rdd.map(lambda x:(Vectors.squared_distance(x[0], centroids[x[1]][1]))).reduce(lambda a, b: a+b)
print(error_df)