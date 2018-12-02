import matplotlib.pyplot as plt
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import KMeans
from pyspark import SparkContext, SparkConf
import datetime as dt
import dateutil.parser as par
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from pyspark.ml.feature import MinMaxScaler
import pyspark.ml.linalg

conf = SparkConf().setAppName("test").setMaster("local[*]")
sc = SparkContext(conf=conf)
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("KMeans") \
    .config("spark.some.config.option", "Angadpreet-KMeans") \
    .getOrCreate()
today = dt.datetime.today()

# Getting the data
spark_df = sc.parallelize(spark.read.json("Data/yelp_academic_dataset_business.json").select("stars","review_count","is_open").take(1700))
scaler = MinMaxScaler(inputCol="_1",\
         outputCol="scaled_1")
trial_df = spark_df.map(lambda x: pyspark.ml.linalg.Vectors.dense(x)).map(lambda x:(x, )).toDF()
scalerModel = scaler.fit(trial_df)
vector_df = scalerModel.transform(trial_df).select("scaled_1").rdd.map(lambda x:Vectors.dense(x))

# Initialize K Means
km = KMeans()
kme = km.train(vector_df, k = 3, maxIterations = 20, initializationMode = "random", seed=2018)
print(kme.computeCost(vector_df))
print(kme.clusterCenters)
df_with = spark.createDataFrame(vector_df.map(lambda x:(float(x[0][0]), float(x[0][1]), float(x[0][2]), 1))).toPandas()
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
scatter = ax.scatter(df_with['_1'],df_with['_2'],df_with['_3'],
                     c=df_with['_4'])
ax.set_title('Business Dataset')
ax.set_xlabel('Stars')
ax.set_ylabel('Review Count')
ax.set_zlabel('Is Open')
cbar = plt.colorbar(scatter)
cbar.remove()
plt.show()