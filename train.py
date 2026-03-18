import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# -----------------------------
# 0. Set HADOOP_HOME for Windows
# -----------------------------
os.environ["HADOOP_HOME"] = "C:\\hadoop"  # Fixed: should point to hadoop root, not bin

# -----------------------------
# 1. Start Spark
# -----------------------------
spark = SparkSession.builder \
    .appName("MusicClassifier") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")  # Suppress warnings in output

# -----------------------------
# 2. Load dataset - FIXED: multiLine + escape for lyrics with newlines
# -----------------------------
df = spark.read \
    .option("header", "true") \
    .option("multiLine", "true") \
    .option("escape", '"') \
    .option("encoding", "UTF-8") \
    .csv("data/Mendeley_dataset.csv")

# -----------------------------
# 3. Debug: print columns to verify
# -----------------------------
print("Columns found:", df.columns)
print(f"Total rows loaded: {df.count()}")

# -----------------------------
# 4. Select and clean needed columns
# -----------------------------
df = df.select("lyrics", "genre")
df = df.dropna(subset=["lyrics", "genre"])

print(f"Rows after dropping nulls: {df.count()}")
print("Genre distribution:")
df.groupBy("genre").count().orderBy("count", ascending=False).show()

# -----------------------------
# 5. Convert genre -> numeric label
# -----------------------------
indexer = StringIndexer(inputCol="genre", outputCol="label", handleInvalid="skip")

# -----------------------------
# 6. Text processing pipeline
# -----------------------------
tokenizer = Tokenizer(inputCol="lyrics", outputCol="words")

# ADDED: StopWordsRemover improves accuracy
remover = StopWordsRemover(inputCol="words", outputCol="filtered")

tf = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=2)

# -----------------------------
# 7. Classifier
# -----------------------------
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20,          # Increased from 10 for better accuracy
    regParam=0.01        # Regularization to prevent overfitting
)

# -----------------------------
# 8. Build Pipeline
# -----------------------------
pipeline = Pipeline(stages=[indexer, tokenizer, remover, tf, idf, lr])

# -----------------------------
# 9. Train/Test split (80/20)
# -----------------------------
train, test = df.randomSplit([0.8, 0.2], seed=42)
print(f"Training rows: {train.count()}, Test rows: {test.count()}")

# -----------------------------
# 10. Train model
# -----------------------------
print("Training model... please wait...")
model = pipeline.fit(train)

# -----------------------------
# 11. Evaluate
# -----------------------------
predictions = model.transform(test)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"\n✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Also show per-genre breakdown
print("\nSample predictions:")
predictions.select("lyrics", "genre", "prediction").show(5, truncate=50)

# -----------------------------
# 12. Save model
# -----------------------------
model.write().overwrite().save("model/music_model")
print("✅ Model trained and saved to model/music_model")

# -----------------------------
# 13. Stop Spark
# -----------------------------
spark.stop()