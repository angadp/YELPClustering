from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import KMeans
from pyspark import SparkContext, SparkConf
import datetime as dt
import dateutil.parser as par
from pyspark.ml.feature import MinMaxScaler
import pyspark.ml.linalg
import sys
from pyspark.sql import SparkSession
import pyspark.mllib.evaluation

# Find the cluster number for the points
def findCenter(vector, centroids):
    dist = sys.maxint
    cluster = -1
    i = 0
    for center in centroids:
        if(Vectors.squared_distance(center, vector)<dist):
            dist = Vectors.squared_distance(center, vector)
            cluster = i
        i += 1
    return cluster

def silhoutte(point, err, num_clusters):
    avg = [0]*num_clusters
    avgi = [0]*num_clusters
    for er in err:
        avg[er[1]] += Vectors.squared_distance(point[0], er[0])
        avgi[er[1]] += 1
    a = avg[point[1]]/avgi[point[1]]
    b = sys.maxint
    for i in range(len(avg)):
        if(i != point[1]):
            if(avg[i]/avgi[i] < b):
                b = avg[i]/avgi[i]
    return (b - a)/max(b, a)

conf = SparkConf().setAppName("test").setMaster("local[*]")
sc = SparkContext(conf=conf)

#Selecting fields and scaling using MinMaxScaler
spark = SparkSession \
    .builder \
    .appName("KMeans") \
    .config("spark.some.config.option", "Angadpreet-KMeans") \
    .getOrCreate()
today = dt.datetime.today()
spark_df = sc.parallelize(spark.read.json("Data/yelp_academic_dataset_business.json").select("stars","review_count","is_open").take(1700))
scaler = MinMaxScaler(inputCol="_1",\
         outputCol="scaled_1")
trial_df = spark_df.map(lambda x: pyspark.ml.linalg.Vectors.dense(x)).map(lambda x:(x, )).toDF()
scalerModel = scaler.fit(trial_df)
vector_df = scalerModel.transform(trial_df).select("scaled_1").rdd.map(lambda x:Vectors.dense(x))
num_clusters = 3

#Input into the Algorithm
km = KMeans()
kme = km.train(vector_df, k = num_clusters, maxIterations = 20, seed=20189080)
centers = kme.clusterCenters

err = vector_df.map(lambda x:(x[0], findCenter(x[0], centers))).collect()

#Silhoutte Value comparison
ag = 0
agi = 0
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
    agi += 1

sil = (ag/agi)

print(sil)

# Number of points in each cluster
df_with = vector_df.map(lambda x:(findCenter(x[0], centers), 1)).reduceByKey(lambda a, b: a+b).collect()
for ji in df_with:
    print(ji)