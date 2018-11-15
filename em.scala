import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{SQLContext, SparkSession}

object em {
  def main(args: Array[String]): Unit = {
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


    val parsedData = df2.map(s => Vectors.dense(s))

    val parsedRdd = sc.parallelize(parsedData)

    // Cluster the data into two classes using GaussianMixture
    val gmm = new GaussianMixture().setK(3).run(parsedRdd)

    // Save and load model
    gmm.save(sc, "target/org/apache/spark/GaussianMixtureExample/GaussianMixtureModel")
    val sameModel = GaussianMixtureModel.load(sc,
      "target/org/apache/spark/GaussianMixtureExample/GaussianMixtureModel")

    // output parameters of max-likelihood model
    for (i <- 0 until gmm.k) {
      println("weight=%f\nmu=%s\nsigma=\n%s\n" format
        (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
    }

  }
}
