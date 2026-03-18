import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# -----------------------------
# 0. Set HADOOP_HOME for Windows (if on Windows)
# -----------------------------
os.environ["HADOOP_HOME"] = "C:\\hadoop\\bin"  # adjust if needed

# -----------------------------
# 1. Start Spark
# -----------------------------
spark = SparkSession.builder \
    .appName("MusicClassifier") \
    .getOrCreate()

# -----------------------------
# 2. Load dataset
# -----------------------------
df = spark.read.csv("data/Mendeley_dataset.csv", header=True, inferSchema=True)

# -----------------------------
# 3. Select needed columns
# -----------------------------
df = df.select("lyrics", "genre")

# Drop rows with missing lyrics or genre
df = df.dropna(subset=["lyrics", "genre"])

# -----------------------------
# 4. Convert genre → numeric label
# -----------------------------
indexer = StringIndexer(inputCol="genre", outputCol="label")

# -----------------------------
# 5. Text processing
# -----------------------------
tokenizer = Tokenizer(inputCol="lyrics", outputCol="words")
tf = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# -----------------------------
# 6. Classifier
# -----------------------------
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

# -----------------------------
# 7. Pipeline
# -----------------------------
pipeline = Pipeline(stages=[indexer, tokenizer, tf, idf, lr])

# -----------------------------
# 8. Train/Test split
# -----------------------------
train, test = df.randomSplit([0.8, 0.2], seed=42)  # seed for reproducibility

# -----------------------------
# 9. Train model
# -----------------------------
model = pipeline.fit(train)

# -----------------------------
# 10. Evaluate (optional)
# -----------------------------
predictions = model.transform(test)
correct = predictions.filter(predictions.label == predictions.prediction).count()
total = predictions.count()
accuracy = correct / total if total != 0 else 0.0
print(f"Model Accuracy: {accuracy:.4f}")

# -----------------------------
# 11. Save model
# -----------------------------
model.write().overwrite().save("model/music_model")

print("✅ Model trained and saved successfully!")

# -----------------------------
# 12. Stop Spark session
# -----------------------------
spark.stop()