# spark_streaming.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

# define schema (same as CSV)
schema = StructType([
    StructField("Time", DoubleType()),
    StructField("V1", DoubleType()), StructField("V2", DoubleType()),
    StructField("V3", DoubleType()), StructField("V4", DoubleType()),
    StructField("V5", DoubleType()), StructField("V6", DoubleType()),
    StructField("V7", DoubleType()), StructField("V8", DoubleType()),
    StructField("V9", DoubleType()), StructField("V10", DoubleType()),
    StructField("V11", DoubleType()), StructField("V12", DoubleType()),
    StructField("V13", DoubleType()), StructField("V14", DoubleType()),
    StructField("V15", DoubleType()), StructField("V16", DoubleType()),
    StructField("V17", DoubleType()), StructField("V18", DoubleType()),
    StructField("V19", DoubleType()), StructField("V20", DoubleType()),
    StructField("V21", DoubleType()), StructField("V22", DoubleType()),
    StructField("V23", DoubleType()), StructField("V24", DoubleType()),
    StructField("V25", DoubleType()), StructField("V26", DoubleType()),
    StructField("V27", DoubleType()), StructField("V28", DoubleType()),
    StructField("Amount", DoubleType()),
    StructField("Class", IntegerType())
])

spark = SparkSession.builder \
    .appName("FraudDetectionStreaming") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.0") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")


df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "transactions") \
    .option("startingOffsets", "earliest") \
    .load()


df_json = df.selectExpr("CAST(value AS STRING) as json")
df_parsed = df_json.select(from_json(col("json"), schema).alias("data")).select("data.*")

query = df_parsed.groupBy("Class").count().writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query.awaitTermination()
spark.stop()