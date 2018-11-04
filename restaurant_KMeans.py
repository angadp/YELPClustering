import matplotlib.pyplot as plt
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import KMeans
from pyspark import SparkContext, SparkConf
import datetime as dt
import dateutil.parser as par

conf = SparkConf().setAppName("test").setMaster("local[*]")
sc = SparkContext(conf=conf)
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("KMeans") \
    .config("spark.some.config.option", "Angadpreet-KMeans") \
    .getOrCreate()
today = dt.datetime.today()
spark_df = spark.read.json("Data/yelp_academic_dataset_business.json").select("review_count", "stars", "is_open").fillna(0).rdd.map(lambda x: (x[0], x[1], x[2]))
vector_df = spark_df.map(lambda s : Vectors.dense(s))
km = KMeans()
kme = km.train(vector_df, 4)
print(kme.computeCost(vector_df))
print(kme.clusterCenters)
