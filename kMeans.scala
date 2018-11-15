import org.apache.log4j.{Level, Logger}
import org.apache.spark
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{SQLContext, SparkSession}



object kMeans {

  def main(args: Array[String]): Unit = {
    // Load and parse the data
    Logger.getLogger("org").setLevel(Level.FATAL)

    val sparkConf = new SparkConf().setAppName("kMeans").setMaster("local[*]")

    val sc = new SparkContext(sparkConf)

    val sqlContext = new SQLContext(sc)


    val spark = SparkSession.builder
      .master("local")
      .appName("Spark JSON Reader")
      .getOrCreate;

    import spark.sqlContext.implicits._

    //val data = sc.textFile("data/mllib/kmeans_data.txt")


    val data = spark.read.json("C://Users/ordaz/OneDrive/Desktop/yelp_dataset/yelp_academic_dataset_user.json")
      .select("review_count", "useful", "average_stars")

    val df3 = data.selectExpr("cast(review_count as double) review_count", "cast(useful as double) useful", "cast(average_stars as double) average_stars"
      )

    df3.printSchema()

    //val df2 = data.selectExpr("cast(review_count as double) review_count", "cast(useful as double) useful")
    val df2 = df3.collect.map(row => (row.getDouble(0)))

    //df2.printSchema()


    //val data_rdd = spark.sparkContext.parallelize(data).toDF("Review", "Useful")
    //val convert = df2.map()


    //val doubledArray = df2.map(_.getDouble(0))

    val parsedData = df2.map(s => Vectors.dense(s))

    val parsedRdd = sc.parallelize(parsedData)


      // Cluster the data into two classes using KMeans
      val numClusters = 3
      val numIterations = 20
      //val clusters = KMeans.train(parsedData, numClusters, numIterations)

      val km = KMeans
      val kme = km.train(parsedRdd,numClusters,numIterations)

      // Evaluate clustering by computing Within Set Sum of Squared Errors
      val WSSSE = kme.computeCost(parsedRdd)
      println(s"Within Set Sum of Squared Errors = $WSSSE")

      // Save and load model
      kme.save(sc, "C://Users/ordaz/OneDrive/Desktop/project_output")
      val sameModel = KMeansModel.load(sc, "C://Users/ordaz/OneDrive/Desktop/project_output")

  }
}
