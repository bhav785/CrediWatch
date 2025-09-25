from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col

# ----------------------------
# Spark Session
# ----------------------------
spark = SparkSession.builder \
    .appName("FraudDetection_RF_HDFS") \
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

# Add weight column
df = df.withColumn("weight", col("Class")*balancing_ratio + 1)

# ----------------------------
# Assemble Features
# ----------------------------
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vector")
scaler = StandardScaler(inputCol="features_vector", outputCol="features_scaled")

# ----------------------------
# Random Forest Classifier
# ----------------------------
rf = RandomForestClassifier(featuresCol="features_scaled",
                            labelCol="Class",
                            weightCol="weight",
                            numTrees=100,
                            maxDepth=5,
                            seed=42)

pipeline = Pipeline(stages=[assembler, scaler, rf])

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
# AUC
evaluator = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="rawPrediction")
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc:.4f}")

# Precision, Recall, F1
multi_eval = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction")
precision = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"})
recall = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"})
f1 = multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# ----------------------------
# Save Model
# ----------------------------
model.write().overwrite().save("hdfs://localhost:9000/user/bhavana/models/fraud_rf_model")

# Stop Spark session
spark.stop()
