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
    .appName("Elbow") \
    .config("spark.some.config.option", "Angadpreet-Elbow") \
    .getOrCreate()
today = dt.datetime.today()
spark_df = spark.read.json("Data/yelp_academic_dataset_user.json").select("review_count", "average_stars", "yelping_since").rdd.map(lambda x: (x[0], x[1], (par.parse(x[2])-today).days))
vector_df = spark_df.map(lambda s : Vectors.dense(s))
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans()
    kme = km.train(vector_df, k)
    Sum_of_squared_distances.append(kme.computeCost(vector_df))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
