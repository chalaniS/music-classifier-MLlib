from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import os
os.environ["HADOOP_HOME"] = "C:/hadoop"
os.environ["PATH"] += ";C:/hadoop/bin"

# ── Spark session ──────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("MusicClassifier-Train") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ── Load data ──────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Merged_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")

print(f"\n📂  Loading data from: {DATA_PATH}")

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("multiLine", "true") \
    .option("escape", '"') \
    .csv(DATA_PATH) \
    .select("lyrics", "genre") \
    .na.drop()

print(f"✅  Loaded {df.count()} rows")
print("🎵  Genre distribution:")
df.groupBy("genre").count().orderBy("count", ascending=False).show()

# ── Pipeline stages ────────────────────────────────────────────────────────────
labelIndexer = StringIndexer(inputCol="genre", outputCol="label", handleInvalid="keep")
tokenizer    = Tokenizer(inputCol="lyrics", outputCol="words")
remover      = StopWordsRemover(inputCol="words", outputCol="filtered")
hashTF       = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=20000)
idf          = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=2)
lr           = LogisticRegression(maxIter=100, regParam=0.01, elasticNetParam=0.0)

pipeline = Pipeline(stages=[labelIndexer, tokenizer, remover, hashTF, idf, lr])

# ── Train / test split ─────────────────────────────────────────────────────────
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"\n🔀  Train: {train_df.count()}  |  Test: {test_df.count()}")

# ── Train ──────────────────────────────────────────────────────────────────────
print("\n🚀  Training model …")
model = pipeline.fit(train_df)

# ── Evaluate ───────────────────────────────────────────────────────────────────
predictions = model.transform(test_df)
evaluator   = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"✅  Test Accuracy: {accuracy * 100:.2f}%")

# ── Save model ─────────────────────────────────────────────────────────────────
if os.path.exists(MODEL_PATH):
    import shutil
    shutil.rmtree(MODEL_PATH)

model.save(MODEL_PATH)
print(f"💾  Model saved to: {MODEL_PATH}")

# Save label-to-genre mapping so the server can decode predictions
labels = model.stages[0].labels          # StringIndexer labels in index order
with open(os.path.join(os.path.dirname(__file__), "model", "labels.txt"), "w") as f:
    f.write("\n".join(labels))

print(f"🏷️   Labels saved: {list(labels)}")
spark.stop()
print("\n🎉  Training complete!\n")