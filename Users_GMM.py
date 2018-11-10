import matplotlib.pyplot as plt
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import GaussianMixture
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
spark_df = spark.read.json("Data/yelp_academic_dataset_user.json").select("review_count", "average_stars").rdd.map(lambda x: (x[0], x[1]))
vector_df = spark_df.map(lambda s : Vectors.dense(s))
gmm = GaussianMixture.train(vector_df, 3)