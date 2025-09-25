from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import pyspark.sql.functions as f

spark = SparkSession.builder \
    .appName("FraudPreprocess_HDFS") \
    .master("local[2]") \
    .getOrCreate()


hdfs_path = "hdfs://localhost:9000/user/bhavana/data/creditcard.csv"
df = spark.read.csv(hdfs_path, header=True, inferSchema=True)


df.select([f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in df.columns]).show()

feature_cols = df.columns
feature_cols.remove("Class")

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vector")


scaler = StandardScaler(inputCol="features_vector", outputCol="features_scaled")


pipeline = Pipeline(stages=[assembler, scaler])
model = pipeline.fit(df)
df_prepared = model.transform(df)

data_final = df_prepared.select("features_scaled", "Class")


train_data, test_data = data_final.randomSplit([0.8, 0.2], seed=42)
print(f"Training rows: {train_data.count()}, Testing rows: {test_data.count()}")


spark.stop()
