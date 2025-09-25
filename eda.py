from pyspark.sql import SparkSession
spark=SparkSession.builder.appName("EDA").getOrCreate()
hdfs_path = "hdfs://localhost:9000/user/bhavana/data/creditcard.csv"
df = spark.read.csv(hdfs_path, header=True, inferSchema=True)
df.printSchema()
df.show(5)
df.describe().show()
print("Total count:",df.count())
df.groupBy("Class").count().show()

