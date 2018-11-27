from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import GaussianMixture
from pyspark import SparkContext, SparkConf
import datetime as dt
import dateutil.parser as par
from pyspark.ml.feature import MinMaxScaler
import pyspark.ml.linalg
import sys

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
vector_df = scalerModel.transform(trial_df).select("scaled_1").rdd.map(lambda x:pyspark.ml.linalg.Vectors.dense(float(x[0][0]), float(x[0][1]), float(x[0][2])))
trial_df = vector_df.map(lambda x: pyspark.ml.linalg.Vectors.dense(x)).map(lambda x:(x, )).toDF(["features"])

# Initialize GMM
gmm = GaussianMixture().setK(4).setMaxIter(60).setSeed(2018)
model = gmm.fit(trial_df)

transformed_df = model.transform(trial_df)  # assign data to gaussian components ("clusters")
err = transformed_df.rdd.map(lambda x: (x[0], x[1])).collect()
num_clusters = 4
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

sil = (ag/agi)

print(sil)
df_with = transformed_df.rdd.map(lambda x:(x[1], 1)).reduceByKey(lambda a, b: a+b).collect()
for ji in df_with:
    print(ji)