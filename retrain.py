import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# -----------------------------
# 0. Set HADOOP_HOME for Windows
# -----------------------------
os.environ["HADOOP_HOME"] = "C:\\hadoop\\bin"

# -----------------------------
# 1. Start Spark
# -----------------------------
spark = SparkSession.builder \
    .appName("MusicClassifierRetrain") \
    .getOrCreate()

# -----------------------------
# 2. Load merged dataset
# -----------------------------
df = spark.read.option("header", "true").csv("data/Merged_dataset.csv")

# Debugging: check column names & schema
df.printSchema()
print("Columns:", df.columns)

# -----------------------------
# 3. Keep only required columns
# -----------------------------
# Ensure column names match exactly what’s in your CSV
# Sometimes CSV columns have spaces or BOM characters
df = df.select("lyrics", "genre")

# -----------------------------
# 4. Drop nulls
# -----------------------------
df = df.dropna(subset=["lyrics", "genre"])  # safer: only drop if relevant columns are null

# -----------------------------
# 5. Convert genre → numeric label
# -----------------------------
label_indexer = StringIndexer(inputCol="genre", outputCol="label")

# -----------------------------
# 6. Text processing
# -----------------------------
tokenizer = Tokenizer(inputCol="lyrics", outputCol="words")
tf = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# -----------------------------
# 7. Classifier
# -----------------------------
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

# -----------------------------
# 8. Pipeline
# -----------------------------
pipeline = Pipeline(stages=[label_indexer, tokenizer, tf, idf, lr])

# -----------------------------
# 9. Train/Test Split
# -----------------------------
train, test = df.randomSplit([0.8, 0.2], seed=42)

# -----------------------------
# 10. Train model
# -----------------------------
model = pipeline.fit(train)

# -----------------------------
# 11. Evaluate (optional)
# -----------------------------
predictions = model.transform(test)

# Calculate accuracy safely
correct = predictions.filter(predictions.label == predictions.prediction).count()
total = predictions.count()
accuracy = correct / total if total != 0 else 0.0

print(f"Model Accuracy: {accuracy:.4f}")

# -----------------------------
# 12. Save NEW model
# -----------------------------
model.write().overwrite().save("model/music_model_8genre")

print("✅ Retrained model saved successfully!")

# -----------------------------
# 13. Stop Spark session (good practice)
# -----------------------------
spark.stop()