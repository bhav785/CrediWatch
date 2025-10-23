# spark_streaming_predict.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.ml import PipelineModel
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# ----------------------------
# Define schema
# ----------------------------
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

# ----------------------------
# Spark Session
# ----------------------------
spark = SparkSession.builder \
    .appName("FraudDetectionStreaming") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.0") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# ----------------------------
# Load trained model
# ----------------------------
model = PipelineModel.load("/mnt/c/Users/bhav0/OneDrive/Documents/Projects/crediwatch/models/logistic_model")

# ----------------------------
# Read from Kafka topic
# ----------------------------
df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "transactions") \
    .option("startingOffsets", "earliest") \
    .load()

df_json = df.selectExpr("CAST(value AS STRING) as json")
df_parsed = df_json.select(from_json(col("json"), schema).alias("data")).select("data.*")

# ----------------------------
# Function to print batch with colors and summary
# ----------------------------
def print_batch(batch_df, batch_id):
    rows = batch_df.collect()
    fraud_count = 0
    normal_count = 0
    
    print(Fore.CYAN + f"\n{'-'*50}\nBatch: {batch_id}\n{'-'*50}")

    for row in rows:
        cls = row['Class']
        pred = int(row['prediction'])
        prob = row['probability'][pred]
        time = row['Time']
        amount = row['Amount']

        if pred == 1:  # Fraud
            fraud_count += 1
            print(Fore.RED + f"FRAUD ALERT! Transaction at time {time}, Amount={amount}, Prediction={pred}, Confidence={prob:.4f}")
        else:  # Normal
            normal_count += 1
            print(Fore.GREEN + f"Normal Transaction at time {time}, Amount={amount}, Prediction={pred}, Confidence={prob:.4f}")
    
    # Mini summary at end of batch
    print(Fore.MAGENTA + f"Batch Summary -> Total Transactions: {len(rows)}, Frauds: {fraud_count}, Normal: {normal_count}")
    print(Fore.CYAN + f"{'-'*50}\n")

# ----------------------------
# Apply model and print
# ----------------------------
predictions = model.transform(df_parsed)

query = predictions.writeStream \
    .foreachBatch(print_batch) \
    .outputMode("append") \
    .start()

query.awaitTermination()
