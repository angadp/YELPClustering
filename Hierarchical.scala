import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection._

object Hierarchical {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.FATAL)

    val sparkConf = new SparkConf().setAppName("kMeans").setMaster("local[*]")

    val sc = new SparkContext(sparkConf)

    val sqlContext = new SQLContext(sc)


    val spark = SparkSession.builder
      .master("local")
      .appName("Spark JSON Reader")
      .getOrCreate;

    import spark.implicits._

    //val data = sc.textFile("data/mllib/kmeans_data.txt")


    val data = spark.read.json("C://Users/ordaz/OneDrive/Desktop/yelp_dataset/yelp_academic_dataset_user.json")
      .select("review_count")



    val df3 = data.selectExpr("cast(review_count as double) features")

    import spark.implicits._



///    val df2 = df3.map(row => row.getString(0)).collect()


    val customSchema = StructType(Array(StructField("review_count", DoubleType, true)))

    val assembler = new VectorAssembler().setInputCols(Array("review_count")).setOutputCol("features")

    val iris_df_trans = assembler.transform(data)

   // val parsedData = df3.map(s => Vectors.dense(s.getDouble(0)))


    //val parsedRdd = sc.parallelize(parsedData)

    //val ssss = dataWithFeatures.


    // Trains a bisecting k-means model.
    val bkm = new BisectingKMeans()
    val model = bkm.fit(iris_df_trans)

    // Evaluate clustering.
    val cost = model.computeCost(iris_df_trans)
    println(s"Within Set Sum of Squared Errors = $cost")

    // Shows the result.
    println("Cluster Centers: ")
    val centers = model.clusterCenters
    centers.foreach(println)


  }
}
