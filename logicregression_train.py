from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col

# ----------------------------
# Spark Session
# ----------------------------
spark = SparkSession.builder \
    .appName("FraudDetection_HDFS") \
    .master("local[2]") \
    .getOrCreate()

# ----------------------------
# Read CSV from HDFS
# ----------------------------
hdfs_path = "hdfs://localhost:9000/user/bhavana/data/creditcard.csv"
df = spark.read.csv(hdfs_path, header=True, inferSchema=True)

# ----------------------------
# Feature Columns
# ----------------------------
feature_cols = df.columns
feature_cols.remove("Class")

# ----------------------------
# Class Imbalance: calculate ratio
# ----------------------------
fraud_count = df.filter(col("Class") == 1).count()
normal_count = df.filter(col("Class") == 0).count()
balancing_ratio = normal_count / fraud_count
print(f"Fraudulent: {fraud_count}, Normal: {normal_count}, Ratio: {balancing_ratio:.2f}")

# ----------------------------
# Assemble Features
# ----------------------------
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vector")
scaler = StandardScaler(inputCol="features_vector", outputCol="features_scaled")

# ----------------------------
# Logistic Regression with weightCol
# ----------------------------
# Add weight column to handle imbalance
df = df.withColumn("weight", col("Class")*balancing_ratio + 1)

lr = LogisticRegression(featuresCol="features_scaled",
                        labelCol="Class",
                        weightCol="weight",
                        maxIter=10)

pipeline = Pipeline(stages=[assembler, scaler, lr])

# ----------------------------
# Train/Test Split
# ----------------------------
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# ----------------------------
# Train Model
# ----------------------------
model = pipeline.fit(train_data)

# ----------------------------
# Predictions
# ----------------------------
predictions = model.transform(test_data)
predictions.select("Class", "prediction", "probability").show(5)

# ----------------------------
# Evaluation
# ----------------------------
evaluator = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="rawPrediction")
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc:.4f}")

# Stop Spark session
spark.stop()
