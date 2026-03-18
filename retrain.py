import os
import pathlib
import chardet
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# -----------------------------
# 0. Set HADOOP_HOME for Windows
# -----------------------------
os.environ["HADOOP_HOME"] = "C:\\hadoop"  # Fixed: root, not bin

# -----------------------------
# 1. PRE-PROCESS CSV with pandas first (fixes encoding + multiline lyrics)
# -----------------------------
csv_path = "data/Merged_dataset.csv"
clean_path = "data/Merged_dataset_clean.csv"

print("🔍 Detecting CSV encoding...")
with open(csv_path, "rb") as f:
    result = chardet.detect(f.read(20000))
detected_enc = result['encoding']
print(f"   Detected: {detected_enc}")

print("🧹 Cleaning CSV...")
df_pd = pd.read_csv(
    csv_path,
    encoding=detected_enc,
    on_bad_lines='skip',
    engine='python'
)

# Strip BOM and spaces from column names
df_pd.columns = df_pd.columns.str.strip().str.replace('\ufeff', '')
print("   Columns found:", df_pd.columns.tolist())

# Keep only required columns
df_pd = df_pd[['artist_name', 'track_name', 'release_date', 'genre', 'lyrics']]

# Clean lyrics - remove newlines and problematic characters
df_pd['lyrics'] = df_pd['lyrics'].astype(str) \
    .str.replace('\n', ' ') \
    .str.replace('\r', ' ') \
    .str.replace('"', "'") \
    .str.strip()

# Drop nulls
df_pd = df_pd.dropna(subset=['lyrics', 'genre'])
df_pd = df_pd[df_pd['lyrics'] != 'nan']

print(f"   Total rows after cleaning: {len(df_pd)}")
print("   Genre distribution:")
print(df_pd['genre'].value_counts())

# Save as clean UTF-8
df_pd.to_csv(clean_path, index=False, encoding='utf-8')
print(f"✅ Clean CSV saved to {clean_path}\n")

# -----------------------------
# 2. Start Spark
# -----------------------------
spark = SparkSession.builder \
    .appName("MusicClassifierRetrain") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# -----------------------------
# 3. Load CLEAN CSV into Spark
# -----------------------------
df = spark.read \
    .option("header", "true") \
    .option("multiLine", "true") \
    .option("escape", '"') \
    .option("encoding", "UTF-8") \
    .csv(clean_path)

print("Columns in Spark:", df.columns)
print(f"Total rows: {df.count()}")

# -----------------------------
# 4. Select and clean
# -----------------------------
df = df.select("lyrics", "genre").dropna(subset=["lyrics", "genre"])
print(f"Rows after dropping nulls: {df.count()}")
print("Genres:")
df.groupBy("genre").count().orderBy("count", ascending=False).show()

# -----------------------------
# 5. Pipeline stages
# -----------------------------
indexer   = StringIndexer(inputCol="genre", outputCol="label", handleInvalid="skip")
tokenizer = Tokenizer(inputCol="lyrics", outputCol="words")
remover   = StopWordsRemover(inputCol="words", outputCol="filtered")
tf        = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
idf       = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=2)
lr        = LogisticRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=20,
                regParam=0.01
            )

pipeline = Pipeline(stages=[indexer, tokenizer, remover, tf, idf, lr])

# -----------------------------
# 6. Train/Test split
# -----------------------------
train, test = df.randomSplit([0.8, 0.2], seed=42)
print(f"Training rows: {train.count()}, Test rows: {test.count()}")

# -----------------------------
# 7. Train
# -----------------------------
print("\n⏳ Training model on 8 genres... please wait...")
model = pipeline.fit(train)

# -----------------------------
# 8. Evaluate
# -----------------------------
predictions = model.transform(test)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"\n✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# -----------------------------
# 9. Save model
# -----------------------------
save_path = pathlib.Path("model/music_model").resolve().as_posix()
print(f"Saving model to: {save_path}")
model.write().overwrite().save(save_path)
print("✅ Retrained 8-genre model saved to model/music_model")

spark.stop()